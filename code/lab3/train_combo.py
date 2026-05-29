"""COMBO: Enhanced features (D1+D2+A2) + GP synthetic data on Split A.

Trains:
  Enhanced + 5000 synth   on Split A train (1449 real + 5000 synth)
  Tests on               Split A test (363 real)

Compare to:
  Big baseline:        1.10 m (single) / 1.08 m (ensemble)
  Big + synth:         0.91 m  ← current best
  Big + Enhanced:      0.95 m
  Combo target:        ?? (<0.91 m hopefully)
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
import synthetic

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
N_SYNTHETIC = 5000


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


def make_synth_records(X_synth, y_synth, bssids):
    """Convert dense synth (X, y) → list of dicts compatible with build_set_input_v2."""
    records = []
    for i in range(len(X_synth)):
        aps = {b: float(X_synth[i, j]) for j, b in enumerate(bssids)
                if X_synth[i, j] > -99.5}
        records.append({
            'x': float(y_synth[i, 0]),
            'y': float(y_synth[i, 1]),
            'session': 'synth',
            'aps': aps,
        })
    return records


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
    return model, hist


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']

    # ── Build real Split A enhanced features ─────────────────────
    print('[build] real enhanced features (D1+D2) ...')
    real_idx, real_feat, real_mask = data.build_set_input_v2(records, bssids,
                                                                max_aps=MAX_APS)

    # ── Generate synthetic from Split A train ────────────────────
    print('\n[gen] fitting GPs on Split A train (1449 records) ...')
    train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                       'session': sess[i],
                       'aps': {b: float(X[i, j])
                                for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                      for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15)

    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)
    free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(SEED)
    if len(free_positions) > N_SYNTHETIC:
        chosen = rng.choice(len(free_positions), N_SYNTHETIC, replace=False)
        positions = free_positions[chosen]
    else:
        positions = free_positions
    print(f'[gen] synthesizing {len(positions)} scans (Fix C, KNN detection prob) ...')
    X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                              records=train_records,
                                              detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0,
                                              seed=SEED)
    aps_per = (X_synth > -99.5).sum(axis=1)
    print(f'[gen] synth APs/scan: mean {aps_per.mean():.1f} (real ~27.7)')

    # Convert synth → enhanced features
    print('[build] synth enhanced features (D1+D2) ...')
    synth_records = make_synth_records(X_synth, y_synth, bssids)
    s_idx, s_feat, s_mask = data.build_set_input_v2(synth_records, bssids,
                                                       max_aps=MAX_APS)

    # ── Build combined training set ──────────────────────────────
    train_idx = np.concatenate([real_idx[tr_A], s_idx], axis=0)
    train_feat = np.concatenate([real_feat[tr_A], s_feat], axis=0)
    train_mask = np.concatenate([real_mask[tr_A], s_mask], axis=0)
    train_y = np.concatenate([y[tr_A], y_synth], axis=0)
    print(f'\n[setup] combined train: {len(train_y)} '
          f'(real {len(tr_A)} + synth {len(s_idx)})')

    train_ds = data.SetDatasetV2(train_idx, train_feat, train_mask, train_y,
                                   jitter=JITTER_DBM, ap_dropout=AP_DROPOUT)
    test_ds = data.SetDatasetV2(real_idx[te_A], real_feat[te_A], real_mask[te_A],
                                  y[te_A], jitter=0.0, ap_dropout=0.0)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(test_ds, batch_size=256, shuffle=False)

    # ── Train COMBO ──────────────────────────────────────────────
    print('\n══════════ COMBO: Big + Enhanced (D1+D2+A2) + 5000 synth ══════════')
    model = models.SetTransformerMDNv2(num_bssids=len(bssids), n_features=3,
                                          **BIG_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[arch] {n_params:,} params')
    model, hist = train_one(model, tl, vl, device, 'A/Combo')

    # Evaluate
    model.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(real_idx[te_A]).to(device)
        f_t = torch.from_numpy(real_feat[te_A]).to(device)
        m_t = torch.from_numpy(real_mask[te_A]).to(device)
        pred = model.predict_xy(idx_t, f_t, m_t, mode='map')
        pi, mu, sigma = model.predict_distribution(idx_t, f_t, m_t)
        nll = float(model.loss(model(idx_t, f_t, m_t),
                                 torch.from_numpy(y[te_A].astype(np.float32)).to(device)).item())
    err = loc_err(pred, y[te_A])
    stats = loc_stats(err)
    print(f'\n══════════ RESULT ══════════')
    print(f'  median={stats["median_err_m"]:.3f}  mean={stats["mean_err_m"]:.3f}'
          f'  p90={stats["p90_err_m"]:.3f}  NLL={nll:.3f}')

    print('\n══════════ Split A progress ladder ══════════')
    print(f'  KNN k=5:                            1.568 m')
    print(f'  Original Set Transformer:           1.093 m')
    print(f'  Big Set Transformer single:         1.100 m')
    print(f'  Big Set Transformer 5-ensemble:     1.083 m')
    print(f'  Big + 5000 synth (no enhanced):     0.906 m')
    print(f'  Big + Enhanced (no synth):          0.950 m')
    print(f'  Big + Enhanced + 5000 synth (this): {stats["median_err_m"]:.3f} m')

    # Save
    torch.save(model.state_dict(),
                CKPT_DIR / 'A_random__SetTransformerCombo.pt')
    pd.DataFrame(hist).to_csv(
        CURVES_DIR / 'A_random__SetTransformerCombo.csv', index=False)
    np.savez(PRED_DIR / 'A_random__SetTransformerCombo.npz',
              pred=pred, y_true=y[te_A], err=err, test_idx=te_A,
              pi=pi, mu=mu, sigma=sigma)

    # Append to metrics.csv
    row = {'split': 'A_random',
            'model': 'SetTransformerCombo',
            **stats, 'nll': nll}
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old['model'] != 'SetTransformerCombo']
        full = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        full = pd.DataFrame([row])
    full.to_csv(csv, index=False)
    print(f'\n[save] -> metrics.csv')


if __name__ == '__main__':
    main()
