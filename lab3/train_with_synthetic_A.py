"""Ablation on Split A: does GP synthetic data help when train/test are same distribution?

Key change from train_with_synthetic.py (which tested on Split C):
  - Synthetic generated from Split A's training records (1449 mixed early+evening)
    → no morning bias, matches Split A test distribution
  - Train and test on Split A directly

Experiments:
  (A) real Split A train (1449)           → Set Transformer Big
  (B) real Split A train + 5000 synth     → Set Transformer Big

Test: Split A test (363 mixed records, never seen)
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

BIG_CFG = dict(embed_dim=48, model_dim=192, num_heads=4,
                num_sab=3, K=5, dropout=0.3)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
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


def train_set(model, train_loader, val_loader, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf'); best_state = None; bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx, val, mask, y in train_loader:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            loss = model.loss(model(idx, val, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in val_loader:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                vloss += float(model.loss(model(idx, val, mask), y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
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
            print(f'  [{tag}] epoch {epoch:3d}  val_loss {vloss:.4f}  best {best_val:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    # Split A
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    print(f'  Split A: train {len(tr_A)}, test {len(te_A)}')

    # ── Fit GPs on Split A train + synthesize ────────────────────
    print('\n[gen] fitting GPs on Split A train (1449 mixed records) ...')
    train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                       'session': sess[i],
                       'aps': {b: X[i, j] for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                      for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15)

    print('[gen] sampling 5000 synthetic positions ...')
    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)
    free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(SEED)
    if len(free_positions) > N_SYNTHETIC:
        chosen = rng.choice(len(free_positions), N_SYNTHETIC, replace=False)
        positions = free_positions[chosen]
    else:
        positions = free_positions

    print(f'[gen] synthesizing {len(positions)} scans ...')
    X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                              records=train_records,
                                              detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0,
                                              seed=SEED)
    aps_per_synth = (X_synth > -99.5).sum(axis=1)
    print(f'[gen] synth APs/scan: mean {aps_per_synth.mean():.1f} '
          f'(real avg ~27.7)')

    s_idx, s_val, s_mask = synthetic.build_set_form(X_synth, bssids, max_aps=MAX_APS)

    # ── Test loader (Split A test) ───────────────────────────────
    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    results = []

    # ── A: real Split A train only ───────────────────────────────
    print('\n══════════ A: real Split A train (1449) ══════════')
    train_ds_A = data.SetDataset(set_idx[tr_A], set_val[tr_A], set_mask[tr_A],
                                   y[tr_A], jitter=JITTER_DBM)
    tl_A = DataLoader(train_ds_A, batch_size=BATCH_SIZE, shuffle=True)
    model_A = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_A, _ = train_set(model_A, tl_A, test_loader, device, 'A_real_only_splitA')
    model_A.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx[te_A]).to(device)
        v_t = torch.from_numpy(set_val[te_A]).to(device)
        m_t = torch.from_numpy(set_mask[te_A]).to(device)
        pred_A = model_A.predict_xy(idx_t, v_t, m_t, mode='map')
    err_A = loc_err(pred_A, y[te_A])
    stats_A = loc_stats(err_A)
    print(f'  RESULT A: median={stats_A["median_err_m"]:.3f}  '
          f'mean={stats_A["mean_err_m"]:.3f}  p90={stats_A["p90_err_m"]:.3f}')
    results.append({'split': 'A_random',
                    'model': 'SetTransformerBig_RealOnly',
                    **stats_A, 'nll': None})

    # ── B: real Split A train + synthetic ────────────────────────
    print('\n══════════ B: real Split A train + 5000 synthetic ══════════')
    combined_idx = np.concatenate([set_idx[tr_A], s_idx], axis=0)
    combined_val = np.concatenate([set_val[tr_A], s_val], axis=0)
    combined_mask = np.concatenate([set_mask[tr_A], s_mask], axis=0)
    combined_y = np.concatenate([y[tr_A], y_synth], axis=0)
    print(f'  train size: {len(combined_y)} (real {len(tr_A)} + synth {len(s_idx)})')

    train_ds_B = data.SetDataset(combined_idx, combined_val, combined_mask,
                                   combined_y, jitter=JITTER_DBM)
    tl_B = DataLoader(train_ds_B, batch_size=BATCH_SIZE, shuffle=True)
    model_B = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_B, _ = train_set(model_B, tl_B, test_loader, device, 'B_real_plus_synth_splitA')
    model_B.eval()
    with torch.no_grad():
        pred_B = model_B.predict_xy(idx_t, v_t, m_t, mode='map')
    err_B = loc_err(pred_B, y[te_A])
    stats_B = loc_stats(err_B)
    print(f'  RESULT B: median={stats_B["median_err_m"]:.3f}  '
          f'mean={stats_B["mean_err_m"]:.3f}  p90={stats_B["p90_err_m"]:.3f}')
    results.append({'split': 'A_random',
                    'model': 'SetTransformerBig_PlusSyntheticA',
                    **stats_B, 'nll': None})

    print('\n══════════ COMPARISON (Split A) ══════════')
    print(f'  A (real only):    median={stats_A["median_err_m"]:.3f}  p90={stats_A["p90_err_m"]:.3f}')
    print(f'  B (real+synth):   median={stats_B["median_err_m"]:.3f}  p90={stats_B["p90_err_m"]:.3f}')
    delta_med = stats_B['median_err_m'] - stats_A['median_err_m']
    delta_p90 = stats_B['p90_err_m'] - stats_A['p90_err_m']
    print(f'  Δ median: {delta_med:+.3f} m ({100 * delta_med / stats_A["median_err_m"]:+.1f}%)')
    print(f'  Δ p90:    {delta_p90:+.3f} m ({100 * delta_p90 / stats_A["p90_err_m"]:+.1f}%)')

    np.savez(PRED_DIR / 'A_random__SetTransformerBig_RealOnly.npz',
              pred=pred_A, y_true=y[te_A], err=err_A, test_idx=te_A)
    np.savez(PRED_DIR / 'A_random__SetTransformerBig_PlusSyntheticA.npz',
              pred=pred_B, y_true=y[te_A], err=err_B, test_idx=te_A)

    # Append to metrics.csv
    new_df = pd.DataFrame(results)
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~old['model'].isin(['SetTransformerBig_RealOnly',
                                         'SetTransformerBig_PlusSyntheticA'])]
        full = pd.concat([old, new_df], ignore_index=True)
    else:
        full = new_df
    full.to_csv(csv, index=False)
    print(f'\n[save] -> metrics.csv (added {len(new_df)} rows)')


if __name__ == '__main__':
    main()
