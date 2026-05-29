"""Ablation: does adding synthetic GP data help cross-time generalization?

Trains BIG Set Transformer MDN (same config as train_ensemble.py) on:
  (A) real morning only            ~912 records           — baseline
  (B) real morning + 5000 synth    ~5912 records          — augmented

Both tested on:
  Split C — real evening (900 records, never seen)

Critical: synthetic data ONLY enters training, NEVER test.

Saves results to metrics.csv as:
  SetTransformerBig_RealOnly      (sanity check, should match prior C result)
  SetTransformerBig_PlusSynthetic  (the experiment)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ── Load real ────────────────────────────────────────────────
    print('[load] real records...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    morning_idx = np.where(sess == 'morning')[0]
    evening_idx = np.where(sess == 'evening')[0]
    print(f'  morning {len(morning_idx)}, evening {len(evening_idx)}')

    # ── Load synthetic ───────────────────────────────────────────
    print('[load] synthetic data...')
    sd = np.load(OUT_DIR / 'synthetic_morning_5000.npz', allow_pickle=True)
    X_synth, y_synth = sd['X_synth'], sd['y_synth']
    print(f'  synthetic: {len(X_synth)}')
    s_idx, s_val, s_mask = synthetic.build_set_form(X_synth, bssids, max_aps=MAX_APS)

    # ── Test set (always real evening) ───────────────────────────
    test_ds = data.SetDataset(set_idx[evening_idx], set_val[evening_idx],
                                set_mask[evening_idx], y[evening_idx], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    results = []

    # ── Experiment A: real morning only ──────────────────────────
    print('\n══════════ A: real morning only (baseline) ══════════')
    tr = morning_idx
    train_ds = data.SetDataset(set_idx[tr], set_val[tr], set_mask[tr],
                                 y[tr], jitter=JITTER_DBM)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    model_A = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_A, _ = train_set(model_A, tl, test_loader, device, 'A_real_only')
    # predict
    model_A.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx[evening_idx]).to(device)
        v_t = torch.from_numpy(set_val[evening_idx]).to(device)
        m_t = torch.from_numpy(set_mask[evening_idx]).to(device)
        pred_A = model_A.predict_xy(idx_t, v_t, m_t, mode='map')
    err_A = loc_err(pred_A, y[evening_idx])
    stats_A = loc_stats(err_A)
    print(f'  RESULT A: median={stats_A["median_err_m"]:.3f}  '
          f'mean={stats_A["mean_err_m"]:.3f}  p90={stats_A["p90_err_m"]:.3f}')
    results.append({'split': 'C_morning_to_evening',
                    'model': 'SetTransformerBig_RealOnly',
                    **stats_A, 'nll': None})
    np.savez(PRED_DIR / 'C_morning_to_evening__SetTransformerBig_RealOnly.npz',
              pred=pred_A, y_true=y[evening_idx], err=err_A, test_idx=evening_idx)

    # ── Experiment B: real morning + synthetic ───────────────────
    print('\n══════════ B: real morning + 5000 synthetic ══════════')
    # Stack real + synthetic into one training set
    combined_idx = np.concatenate([set_idx[morning_idx], s_idx], axis=0)
    combined_val = np.concatenate([set_val[morning_idx], s_val], axis=0)
    combined_mask = np.concatenate([set_mask[morning_idx], s_mask], axis=0)
    combined_y = np.concatenate([y[morning_idx], y_synth], axis=0)
    print(f'  train size: {len(combined_y)} (real {len(morning_idx)} + synth {len(s_idx)})')

    train_ds_B = data.SetDataset(combined_idx, combined_val, combined_mask,
                                   combined_y, jitter=JITTER_DBM)
    tl_B = DataLoader(train_ds_B, batch_size=BATCH_SIZE, shuffle=True)
    model_B = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_B, _ = train_set(model_B, tl_B, test_loader, device, 'B_real_plus_synth')
    model_B.eval()
    with torch.no_grad():
        pred_B = model_B.predict_xy(idx_t, v_t, m_t, mode='map')
    err_B = loc_err(pred_B, y[evening_idx])
    stats_B = loc_stats(err_B)
    print(f'  RESULT B: median={stats_B["median_err_m"]:.3f}  '
          f'mean={stats_B["mean_err_m"]:.3f}  p90={stats_B["p90_err_m"]:.3f}')
    results.append({'split': 'C_morning_to_evening',
                    'model': 'SetTransformerBig_PlusSynthetic',
                    **stats_B, 'nll': None})
    np.savez(PRED_DIR / 'C_morning_to_evening__SetTransformerBig_PlusSynthetic.npz',
              pred=pred_B, y_true=y[evening_idx], err=err_B, test_idx=evening_idx)

    # ── Compare + save ───────────────────────────────────────────
    print('\n══════════ COMPARISON ══════════')
    print(f'  A (real only):    median={stats_A["median_err_m"]:.3f}  p90={stats_A["p90_err_m"]:.3f}')
    print(f'  B (real+synth):   median={stats_B["median_err_m"]:.3f}  p90={stats_B["p90_err_m"]:.3f}')
    delta_med = stats_B['median_err_m'] - stats_A['median_err_m']
    delta_p90 = stats_B['p90_err_m'] - stats_A['p90_err_m']
    print(f'  Δ median: {delta_med:+.3f} m ({100 * delta_med / stats_A["median_err_m"]:+.1f}%)')
    print(f'  Δ p90:    {delta_p90:+.3f} m ({100 * delta_p90 / stats_A["p90_err_m"]:+.1f}%)')

    # Append to existing metrics.csv
    new_df = pd.DataFrame(results)
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        # remove any prior rows with these model names (clean re-runs)
        old = old[~old['model'].isin(['SetTransformerBig_RealOnly',
                                         'SetTransformerBig_PlusSynthetic'])]
        full = pd.concat([old, new_df], ignore_index=True)
    else:
        full = new_df
    full.to_csv(csv, index=False)
    print(f'\n[save] -> metrics.csv (added {len(new_df)} rows)')


if __name__ == '__main__':
    main()
