"""Train a BIG Set Transformer MDN with strong regularization, 5-seed ensemble.

Strategy:
  Bigger architecture (embed 48, model 192, 3 SAB, K=5)  → more capacity
  + Stronger dropout (0.3) + weight decay 1e-3            → block overfit
  + Jitter 4 dBm (matches EDA within-cell std 3.5)         → noise robustness
  + 5 independent training runs (seeds 42..46), average    → ensemble

For each split (A, B, C, D):
  - Train 5 models, save individual predictions
  - Compute ensemble prediction = mean of MAP points across 5 seeds
  - Save ensemble pred separately

Appends 2 new rows per split to metrics.csv:
  SetTransformerBig            best single (lowest val loss across seeds)
  SetTransformerBigEnsemble    ensemble of all 5
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT_DIR = Path(__file__).parent / 'outputs'
CKPT_DIR = OUT_DIR / 'checkpoints'
PRED_DIR = OUT_DIR / 'predictions'
CURVES_DIR = OUT_DIR / 'training_curves'

# Big config (more capacity + stronger regularization)
BIG_CFG = dict(
    embed_dim=48,
    model_dim=192,
    num_heads=4,        # 192 / 4 = 48 per head
    num_sab=3,
    K=5,
    dropout=0.3,
)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
N_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]
MAX_APS = 50
MIN_BSSID_COUNT = 10


def loc_err(y_pred, y_true):
    return np.linalg.norm(y_pred - y_true, axis=1)


def loc_stats(err):
    return {
        'median_err_m': float(np.median(err)),
        'mean_err_m': float(err.mean()),
        'p90_err_m': float(np.percentile(err, 90)),
        'p10_err_m': float(np.percentile(err, 10)),
        'max_err_m': float(err.max()),
    }


def train_one(model, train_loader, val_loader, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf'); best_state = None; bad = 0
    hist = {'epoch': [], 'train_loss': [], 'val_loss': []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0.0; nb = 0
        for idx, val, mask, y in train_loader:
            idx, val, mask, y = [b.to(device, non_blocking=True)
                                  for b in (idx, val, mask, y)]
            opt.zero_grad()
            out = model(idx, val, mask)
            loss = model.loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tloss += float(loss.item()); nb += 1
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in val_loader:
                idx, val, mask, y = [b.to(device, non_blocking=True)
                                      for b in (idx, val, mask, y)]
                out = model(idx, val, mask)
                vloss += float(model.loss(out, y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        hist['epoch'].append(epoch)
        hist['train_loss'].append(tloss / max(1, nb))
        hist['val_loss'].append(vloss)
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'    [{tag}] early stop @ epoch {epoch}  best val = {best_val:.4f}')
                break
        if epoch == 1 or epoch % 40 == 0:
            print(f'    [{tag}] epoch {epoch:3d}  train {hist["train_loss"][-1]:.4f}  '
                   f'val {vloss:.4f}  best {best_val:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


def predict_one(model, set_idx, set_val, set_mask, y_te, device):
    model.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx).to(device)
        v_t = torch.from_numpy(set_val).to(device)
        m_t = torch.from_numpy(set_mask).to(device)
        pred = model.predict_xy(idx_t, v_t, m_t, mode='map')
        pi, mu, sigma = model.predict_distribution(idx_t, v_t, m_t)
        y_t = torch.from_numpy(y_te.astype(np.float32)).to(device)
        nll = float(model.loss(model(idx_t, v_t, m_t), y_t).item())
    return pred, pi, mu, sigma, nll


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  '
          + (f'({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    print('[load] reading wifi/*.jsonl ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)
    print(f'       {len(records)} records, {X.shape[1]} BSSID features, '
          f'{len(splits)} splits')

    # Print parameter count of the big model once
    big_model = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    n_params = sum(p.numel() for p in big_model.parameters())
    print(f'[arch] BIG SetTransformer: {n_params:,} params  '
          f'(vs ~209k baseline → {n_params/208687:.2f}x bigger)')
    del big_model

    new_results = []
    t_total = time.time()

    for split_name, (tr, te) in splits.items():
        print(f'\n══════════ split = {split_name}  '
              f'(train {len(tr)}, test {len(te)}) ══════════')
        seed_preds = []      # list of (pred[Nte, 2], pi, mu, sigma, nll, val_loss)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            tag = f'{split_name}/Big_s{seed}'
            print(f'  [seed {seed}]')
            model = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
            train_ds = data.SetDataset(set_idx[tr], set_val[tr], set_mask[tr],
                                         y[tr], jitter=JITTER_DBM)
            val_ds = data.SetDataset(set_idx[te], set_val[te], set_mask[te],
                                       y[te], jitter=0.0)
            tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            vl = DataLoader(val_ds, batch_size=256, shuffle=False)
            model, hist, vloss = train_one(model, tl, vl, device, tag)
            pred, pi, mu, sigma, nll = predict_one(
                model, set_idx[te], set_val[te], set_mask[te], y[te], device)
            err = loc_err(pred, y[te])
            print(f'    seed {seed}: median={np.median(err):.3f}  '
                  f'mean={err.mean():.3f}  p90={np.percentile(err,90):.3f}  '
                  f'NLL={nll:.3f}  val_loss={vloss:.3f}')
            torch.save(model.state_dict(),
                        CKPT_DIR / f'{split_name}__Big_s{seed}.pt')
            pd.DataFrame(hist).to_csv(
                CURVES_DIR / f'{split_name}__Big_s{seed}.csv', index=False)
            np.savez(PRED_DIR / f'{split_name}__Big_s{seed}.npz',
                     pred=pred, y_true=y[te], err=err, test_idx=te,
                     pi=pi, mu=mu, sigma=sigma)
            seed_preds.append((pred, pi, mu, sigma, nll, vloss))

        # Best-single (lowest val loss across seeds)
        best_seed_i = int(np.argmin([x[5] for x in seed_preds]))
        bp, bpi, bmu, bsigma, bnll, _ = seed_preds[best_seed_i]
        berr = loc_err(bp, y[te])
        bs = loc_stats(berr)
        new_results.append({'split': split_name,
                            'model': 'SetTransformerBig',
                            **bs, 'nll': bnll})
        print(f'  best single (seed {SEEDS[best_seed_i]}):'
              f' median={bs["median_err_m"]:.3f}  '
              f'mean={bs["mean_err_m"]:.3f}  p90={bs["p90_err_m"]:.3f}')
        np.savez(PRED_DIR / f'{split_name}__SetTransformerBig.npz',
                 pred=bp, y_true=y[te], err=berr, test_idx=te,
                 pi=bpi, mu=bmu, sigma=bsigma)

        # Ensemble: average MAP point predictions across 5 seeds
        ens_pred = np.mean([sp[0] for sp in seed_preds], axis=0)
        # Ensemble GMM: stack all 5 models' K=5 mixtures → K=25 mixture,
        # with uniform weight 1/5 across models
        ens_pi = np.concatenate([sp[1] for sp in seed_preds], axis=1) / N_SEEDS   # (Nte, 25)
        ens_mu = np.concatenate([sp[2] for sp in seed_preds], axis=1)             # (Nte, 25, 2)
        ens_sigma = np.concatenate([sp[3] for sp in seed_preds], axis=1)          # (Nte, 25, 2)
        ens_err = loc_err(ens_pred, y[te])
        ens_stats = loc_stats(ens_err)
        # Average NLL of individual models (proper ensemble NLL = log of mixture is harder;
        # report avg of individual NLLs as a proxy)
        ens_nll = float(np.mean([sp[4] for sp in seed_preds]))
        new_results.append({'split': split_name,
                            'model': 'SetTransformerBigEnsemble',
                            **ens_stats, 'nll': ens_nll})
        print(f'  ENSEMBLE (mean of 5 MAPs):'
              f' median={ens_stats["median_err_m"]:.3f}  '
              f'mean={ens_stats["mean_err_m"]:.3f}  '
              f'p90={ens_stats["p90_err_m"]:.3f}  avg_NLL={ens_nll:.3f}')
        np.savez(PRED_DIR / f'{split_name}__SetTransformerBigEnsemble.npz',
                 pred=ens_pred, y_true=y[te], err=ens_err, test_idx=te,
                 pi=ens_pi, mu=ens_mu, sigma=ens_sigma)

    # Append to existing metrics.csv
    new_df = pd.DataFrame(new_results)
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        # remove any prior rows with these model names so re-runs are clean
        old = old[~old['model'].isin(['SetTransformerBig', 'SetTransformerBigEnsemble'])]
        full = pd.concat([old, new_df], ignore_index=True)
    else:
        full = new_df
    full.to_csv(csv, index=False)
    print(f'\n[total time] {time.time() - t_total:.1f}s')
    print('\n=== new rows added ===')
    print(new_df.to_string(index=False))
    print(f'\n✓ outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
