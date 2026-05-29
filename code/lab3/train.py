"""Train all models on all 3 splits.

Models:
  KNN_k1, KNN_k5          classical fingerprinting baseline
  MLP                     vec input, Huber loss            (current SOTA-MLP)
  MDN                     vec input, NLL loss              (probabilistic)
  MaskedMLP               (rssi, presence_mask) input      (zero-cost upgrade)
  MaskedMDN               (rssi, presence_mask) input + MDN
  SetTransformerMDN       set-of-(BSSID_emb, rssi) input + MDN  (2025 SOTA)

Saves to outputs/:
  metrics.csv             per (model, split) localization stats + NLL
  predictions/            per-sample predictions (npz)
  checkpoints/            model state dicts
  training_curves/        train/val loss per epoch
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

EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40
JITTER_DBM = 2.0           # train-time RSSI augmentation
MIN_BSSID_COUNT = 10       # BSSID kept as feature if ≥10 records
MAX_APS = 50               # for Set Transformer; max set size per scan


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# ─────────────────────────────────────────────────────────────────────
# Generic train loop — handles 2/3/4-tuple batches via len(batch)
# ─────────────────────────────────────────────────────────────────────

def batch_to_device(batch, device):
    return tuple(b.to(device, non_blocking=True) for b in batch)


def model_forward(model, batch):
    """Returns (out, target).  Batch comes already on device."""
    if len(batch) == 2:           # (x, y)
        x, y = batch
        return model(x), y
    if len(batch) == 3:           # (x, mask, y)
        x, m, y = batch
        return model(x, m), y
    if len(batch) == 4:           # (idx, val, mask, y)
        idx, v, m, y = batch
        return model(idx, v, m), y
    raise ValueError(f'unexpected batch tuple of len {len(batch)}')


def train_loop(model, train_loader, val_loader, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_val = float('inf'); best_state = None; bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0.0; nb = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad()
            out, y = model_forward(model, batch)
            loss = model.loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tloss += float(loss.item()); nb += 1
        sched.step()

        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                out, y = model_forward(model, batch)
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


# ─────────────────────────────────────────────────────────────────────
# Per-model-family run helpers
# ─────────────────────────────────────────────────────────────────────

def run_vec_model(model_cls, in_dim, X, y, tr, te, device, tag):
    """Train a vec-input model (MLP / MDN with -100 padding)."""
    model = model_cls(in_dim=in_dim)
    tl, vl = data.make_loaders(X[tr], y[tr], X[te], y[te],
                                 batch_size=BATCH_SIZE, jitter_dBm=JITTER_DBM, mode='vec')
    model, hist, _ = train_loop(model, tl, vl, device, tag)
    # predict
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(data.normalize_X(X[te])).to(device)
        if isinstance(model, models.MDNRegressor):
            pred = model.predict_xy(x_t, mode='map')
            pi, mu, sigma = model.predict_distribution(x_t)
            nll = float(model.loss(model(x_t),
                                    torch.from_numpy(y[te].astype(np.float32)).to(device)).item())
            extras = {'pi': pi, 'mu': mu, 'sigma': sigma, 'nll': nll}
        else:
            pred = model.predict_xy(x_t)
            extras = {}
    return model, hist, pred, extras


def run_masked_model(model_cls, in_dim, X, y, tr, te, device, tag):
    model = model_cls(in_dim=in_dim)
    tl, vl = data.make_loaders(X[tr], y[tr], X[te], y[te],
                                 batch_size=BATCH_SIZE, jitter_dBm=JITTER_DBM, mode='masked')
    model, hist, _ = train_loop(model, tl, vl, device, tag)
    model.eval()
    with torch.no_grad():
        x_norm = data.normalize_X(X[te])
        x_t = torch.from_numpy(x_norm).to(device)
        m_t = torch.from_numpy(data.build_mask(X[te])).to(device)
        if isinstance(model, models.MaskedMDN):
            pred = model.predict_xy(x_t, m_t, mode='map')
            pi, mu, sigma = model.predict_distribution(x_t, m_t)
            y_t = torch.from_numpy(y[te].astype(np.float32)).to(device)
            nll = float(model.loss(model(x_t, m_t), y_t).item())
            extras = {'pi': pi, 'mu': mu, 'sigma': sigma, 'nll': nll}
        else:
            pred = model.predict_xy(x_t, m_t)
            extras = {}
    return model, hist, pred, extras


def run_set_model(num_bssids, set_idx, set_val, set_mask, y,
                   tr, te, device, tag):
    model = models.SetTransformerMDN(num_bssids=num_bssids,
                                       embed_dim=32, model_dim=128,
                                       num_heads=4, num_sab=2, K=3)
    train_ds = data.SetDataset(set_idx[tr], set_val[tr], set_mask[tr], y[tr],
                                jitter=JITTER_DBM)
    val_ds = data.SetDataset(set_idx[te], set_val[te], set_mask[te], y[te], jitter=0.0)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(val_ds, batch_size=256, shuffle=False)
    model, hist, _ = train_loop(model, tl, vl, device, tag)

    model.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx[te]).to(device)
        v_t = torch.from_numpy(set_val[te]).to(device)
        m_t = torch.from_numpy(set_mask[te]).to(device)
        pred = model.predict_xy(idx_t, v_t, m_t, mode='map')
        pi, mu, sigma = model.predict_distribution(idx_t, v_t, m_t)
        y_t = torch.from_numpy(y[te].astype(np.float32)).to(device)
        nll = float(model.loss(model(idx_t, v_t, m_t), y_t).item())
    return model, hist, pred, {'pi': pi, 'mu': mu, 'sigma': sigma, 'nll': nll}


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f'device: {device}  '
          + (f'({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    torch.manual_seed(42); np.random.seed(42)

    print('[load] reading wifi/*.jsonl ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    print(f'       {len(records)} records, {X.shape[1]} BSSID features')

    print('[set]  building set-form inputs ...')
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    # Persist artifacts for evaluate.py
    np.save(OUT_DIR / 'X.npy', X)
    np.save(OUT_DIR / 'y.npy', y)
    np.save(OUT_DIR / 'sess.npy', sess)
    (OUT_DIR / 'bssids.json').write_text(json.dumps(bssids, indent=2), encoding='utf-8')
    with open(OUT_DIR / 'splits.json', 'w', encoding='utf-8') as f:
        json.dump({k: {'train': v[0].tolist(), 'test': v[1].tolist()}
                    for k, v in splits.items()}, f)

    results = []
    in_dim = X.shape[1]

    for split_name, (tr, te) in splits.items():
        print(f'\n══════════ split = {split_name}  '
              f'(train {len(tr)}, test {len(te)}) ══════════')

        # KNN baselines
        for k in (1, 5):
            knn = models.KNNRegressor(k=k).fit(X[tr], y[tr])
            pred = knn.predict(X[te])
            err = loc_err(pred, y[te])
            s = loc_stats(err)
            results.append({'split': split_name, 'model': f'KNN_k{k}', **s, 'nll': None})
            print(f'  KNN_k{k}  median={s["median_err_m"]:.3f}  '
                  f'mean={s["mean_err_m"]:.3f}  p90={s["p90_err_m"]:.3f}')
            np.savez(PRED_DIR / f'{split_name}__KNN_k{k}.npz',
                     pred=pred, y_true=y[te], err=err, test_idx=te)

        # Torch models
        runs = [
            ('MLP', lambda: run_vec_model(models.MLPRegressor, in_dim, X, y, tr, te, device, f'{split_name}/MLP')),
            ('MDN', lambda: run_vec_model(models.MDNRegressor, in_dim, X, y, tr, te, device, f'{split_name}/MDN')),
            ('MaskedMLP', lambda: run_masked_model(models.MaskedMLP, in_dim, X, y, tr, te, device, f'{split_name}/MaskedMLP')),
            ('MaskedMDN', lambda: run_masked_model(models.MaskedMDN, in_dim, X, y, tr, te, device, f'{split_name}/MaskedMDN')),
            ('SetTransformerMDN', lambda: run_set_model(len(bssids), set_idx, set_val, set_mask, y, tr, te, device, f'{split_name}/SetTrans')),
        ]
        for name, fn in runs:
            print(f'  [train {name}]')
            model, hist, pred, extras = fn()
            err = loc_err(pred, y[te])
            s = loc_stats(err)
            row = {'split': split_name, 'model': name, **s, 'nll': extras.get('nll')}
            results.append(row)
            print(f'  {name:18s}  median={s["median_err_m"]:.3f}  '
                  f'mean={s["mean_err_m"]:.3f}  p90={s["p90_err_m"]:.3f}'
                  + (f'  NLL={extras["nll"]:.3f}' if 'nll' in extras else ''))
            torch.save(model.state_dict(), CKPT_DIR / f'{split_name}__{name}.pt')
            pd.DataFrame(hist).to_csv(CURVES_DIR / f'{split_name}__{name}.csv', index=False)
            save_kw = dict(pred=pred, y_true=y[te], err=err, test_idx=te)
            if 'pi' in extras:
                save_kw.update(pi=extras['pi'], mu=extras['mu'], sigma=extras['sigma'])
            np.savez(PRED_DIR / f'{split_name}__{name}.npz', **save_kw)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / 'metrics.csv', index=False)
    print('\n=== final metrics ===')
    print(df.to_string(index=False))
    print(f'\nall outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
