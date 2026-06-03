"""Train Set Transformer with D1+D2+A2 enhancements:
  D1: per-AP rank feature       (in [0, 1], strongest = 1)
  D2: relative RSSI feature     (rel to strongest in scan, in [-2, 0])
  A2: AP dropout augmentation   (10% real APs masked at training time)

Tests on Splits A and C (the two we care about).
Compares to Big Set Transformer baseline (no enhancements):
  Split A: 1.10 m (single) / 1.08 m (ensemble)
  Split C: 1.73 m (single) / 1.85 m (ensemble)
"""
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
PRED_DIR = OUT_DIR / 'predictions'
CKPT_DIR = OUT_DIR / 'checkpoints'
CURVES_DIR = OUT_DIR / 'training_curves'

BIG_CFG = dict(embed_dim=48, model_dim=192, num_heads=4,
                num_sab=3, K=5, dropout=0.3)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
AP_DROPOUT = 0.10
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
SEED = 42
MAX_APS = 50
MIN_BSSID_COUNT = 10
SPLITS_TO_RUN = ['A_random', 'C_morning_to_evening', 'D_stratified']


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
        for idx, feat, mask, y in train_loader:
            idx = idx.to(device); feat = feat.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            loss = model.loss(model(idx, feat, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tloss += float(loss.item()); nb += 1
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, feat, mask, y in val_loader:
                idx = idx.to(device); feat = feat.to(device)
                mask = mask.to(device); y = y.to(device)
                vloss += float(model.loss(model(idx, feat, mask), y).item()) * y.size(0)
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
                print(f'  [{tag}] early stop @ epoch {epoch}, best val = {best_val:.4f}')
                break
        if epoch == 1 or epoch % 30 == 0:
            print(f'  [{tag}] epoch {epoch:3d}  train {hist["train_loss"][-1]:.4f}  '
                  f'val {vloss:.4f}  best {best_val:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)

    print('[build] enhanced set features (D1+D2) ...')
    set_idx, feat3, mask = data.build_set_input_v2(records, bssids, max_aps=MAX_APS)
    print(f'  feature tensor shape: {feat3.shape}  (3 channels: rssi, rank, rel_rssi)')

    # Sanity: print param count of enhanced model
    enh = models.SetTransformerMDNv2(num_bssids=len(bssids), n_features=3, **BIG_CFG)
    n_params = sum(p.numel() for p in enh.parameters())
    print(f'[arch] SetTransformerMDNv2 (D1+D2 features): {n_params:,} params  '
          f'(vs ~619k baseline)')
    del enh

    new_results = []
    for split_name in SPLITS_TO_RUN:
        tr, te = splits[split_name]
        print(f'\n══════════ split = {split_name}  '
              f'(train {len(tr)}, test {len(te)}) ══════════')

        train_ds = data.SetDatasetV2(set_idx[tr], feat3[tr], mask[tr], y[tr],
                                       jitter=JITTER_DBM, ap_dropout=AP_DROPOUT)
        test_ds = data.SetDatasetV2(set_idx[te], feat3[te], mask[te], y[te],
                                      jitter=0.0, ap_dropout=0.0)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = models.SetTransformerMDNv2(num_bssids=len(bssids), n_features=3,
                                              **BIG_CFG)
        model, hist, _ = train_one(model, tl, vl, device, f'{split_name}/Enh')

        # Evaluate
        model.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te]).to(device)
            f_t = torch.from_numpy(feat3[te]).to(device)
            m_t = torch.from_numpy(mask[te]).to(device)
            pred = model.predict_xy(idx_t, f_t, m_t, mode='map')
            pi, mu, sigma = model.predict_distribution(idx_t, f_t, m_t)
            nll = float(model.loss(model(idx_t, f_t, m_t),
                                     torch.from_numpy(y[te].astype(np.float32)).to(device)).item())
        err = loc_err(pred, y[te])
        stats = loc_stats(err)
        print(f'  RESULT: median={stats["median_err_m"]:.3f}  mean={stats["mean_err_m"]:.3f}'
              f'  p90={stats["p90_err_m"]:.3f}  NLL={nll:.3f}')

        new_results.append({'split': split_name,
                            'model': 'SetTransformerEnhanced',
                            **stats, 'nll': nll})
        torch.save(model.state_dict(),
                    CKPT_DIR / f'{split_name}__SetTransformerEnhanced.pt')
        pd.DataFrame(hist).to_csv(
            CURVES_DIR / f'{split_name}__SetTransformerEnhanced.csv', index=False)
        np.savez(PRED_DIR / f'{split_name}__SetTransformerEnhanced.npz',
                  pred=pred, y_true=y[te], err=err, test_idx=te,
                  pi=pi, mu=mu, sigma=sigma)

    # Append to metrics.csv
    new_df = pd.DataFrame(new_results)
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old['model'] != 'SetTransformerEnhanced']
        full = pd.concat([old, new_df], ignore_index=True)
    else:
        full = new_df
    full.to_csv(csv, index=False)
    print('\n=== ENHANCED RESULTS ===')
    print(new_df.to_string(index=False))
    print('\nvs baselines:')
    print('  A_random:           Big single 1.10 m,  ensemble 1.08 m')
    print('  C_morning_to_evening: Big single 1.73 m,  ensemble 1.85 m')
    print('  D_stratified:       Big single 0.98 m,  ensemble 1.03 m')


if __name__ == '__main__':
    main()
