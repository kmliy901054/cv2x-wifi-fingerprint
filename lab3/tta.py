"""Test-Time Augmentation (TTA) for Set Transformer MDN.

Strategy:
  For each test scan, run model N times with different input perturbations:
    - RSSI jitter ±2 dBm
    - AP dropout 5%
  Aggregate N predictions into one final estimate.

Aggregation strategies tried:
  - Mean of MAP points         (simplest)
  - GMM mixture                (concat all components, reweight by 1/N)
  - Weighted mean by argmax-π   (use model's own confidence)

Tests on Split A using checkpoints from train_best_ensemble.py:
  A_random__BestCombo_s42.pt ... A_random__BestCombo_s46.pt
  (Big Set Transformer + 5000 GP synth, 5-seed ensemble baseline 0.889 m)
Then averages across the 5 seeds for an ensemble + TTA result.

Reference: arxiv 2409.12587 (TTA Meets Variational Bayes 2024).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT_DIR = Path(__file__).parent / 'outputs'
CKPT_DIR = OUT_DIR / 'checkpoints'
PRED_DIR = OUT_DIR / 'predictions'

BIG_CFG = dict(embed_dim=48, model_dim=192, num_heads=4,
                num_sab=3, K=5, dropout=0.3)
MAX_APS = 50
MIN_BSSID_COUNT = 10
SEEDS = [42, 43, 44, 45, 46]

N_TTA = 50                # number of TTA samples per test point
TTA_JITTER_DBM = 1.0      # small jitter — model already robust to noise
TTA_AP_DROPOUT = 0.0      # 0: model never trained w/ AP dropout → don't apply at test
AGGREGATE = 'median'      # 'mean' or 'median' (median = Weiszfeld geometric median)


def loc_err(y_pred, y_true):
    return np.linalg.norm(y_pred - y_true, axis=1)


def loc_stats(err):
    return {
        'median_err_m': float(np.median(err)),
        'mean_err_m': float(err.mean()),
        'p90_err_m': float(np.percentile(err, 90)),
        'p10_err_m': float(np.percentile(err, 10)),
    }


def augment_batch(idx, val, mask, jitter_norm, ap_dropout, rng):
    """Apply TTA augmentations.  All tensors stay on the same device."""
    # RSSI jitter (only on real entries)
    if jitter_norm > 0:
        noise = (torch.rand_like(val) * 2 - 1) * jitter_norm
        val = val + noise * mask
    # AP dropout
    if ap_dropout > 0:
        keep_prob = 1.0 - ap_dropout
        keep = (torch.rand_like(mask) < keep_prob).float()
        mask = mask * keep
    return idx, val, mask


@torch.no_grad()
def tta_predict(model, set_idx_te, set_val_te, set_mask_te, device,
                  n_tta=N_TTA, jitter=TTA_JITTER_DBM, ap_dropout=TTA_AP_DROPOUT,
                  seed=42):
    """Run N_TTA augmented forwards, return mean-MAP predictions + GMM aggregates."""
    model.eval().to(device)
    jitter_norm = jitter / 20.0
    rng = torch.Generator(device=device).manual_seed(seed)
    idx_t = torch.from_numpy(set_idx_te).to(device)
    val_t = torch.from_numpy(set_val_te).to(device)
    mask_t = torch.from_numpy(set_mask_te).to(device)

    N = idx_t.size(0)
    all_maps = []   # (N_TTA, N, 2)
    all_pi = []     # (N_TTA, N, K)
    all_mu = []     # (N_TTA, N, K, 2)
    all_sigma = []  # (N_TTA, N, K, 2)

    for n in range(n_tta):
        torch.manual_seed(seed + n * 1000)
        aug_idx, aug_val, aug_mask = augment_batch(idx_t, val_t, mask_t,
                                                      jitter_norm, ap_dropout, rng)
        log_pi, mu, log_sigma = model(aug_idx, aug_val, aug_mask)
        k = log_pi.argmax(dim=-1)
        map_pts = mu[torch.arange(N), k]   # (N, 2)
        all_maps.append(map_pts.cpu().numpy())
        all_pi.append(log_pi.exp().cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_sigma.append(log_sigma.exp().cpu().numpy())

    all_maps = np.stack(all_maps, axis=0)          # (N_TTA, N, 2)
    if AGGREGATE == 'median':
        # Geometric median via Weiszfeld iteration
        agg_map = geometric_median(all_maps)
    else:
        agg_map = all_maps.mean(axis=0)
    # TTA uncertainty proxy: spread of MAP predictions
    map_std = all_maps.std(axis=0)                  # (N, 2)
    map_unc = np.linalg.norm(map_std, axis=1)       # (N,)

    return agg_map, map_unc, all_maps, all_pi, all_mu, all_sigma


def geometric_median(points, eps=1e-5, max_iter=100):
    """Weiszfeld's algorithm — geometric median for each test sample.
    points: (N_TTA, N, 2)  → return (N, 2)
    """
    median = points.mean(axis=0)        # init with mean
    for _ in range(max_iter):
        d = np.linalg.norm(points - median[None, :, :], axis=-1)  # (N_TTA, N)
        d = np.clip(d, eps, None)
        w = 1.0 / d                                                # (N_TTA, N)
        w_sum = w.sum(axis=0, keepdims=True)                       # (1, N)
        new_median = (w[:, :, None] * points).sum(axis=0) / w_sum.T
        if np.max(np.linalg.norm(new_median - median, axis=-1)) < eps:
            break
        median = new_median
    return median


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)
    tr_A, te_A = splits['A_random']

    y_te = y[te_A]
    print(f'  test set: {len(te_A)} samples (Split A test)')

    # ── Load each seed's checkpoint, baseline + TTA ────────────
    print(f'\nEvaluating with TTA: N_TTA={N_TTA}, jitter=±{TTA_JITTER_DBM} dBm, '
          f'AP dropout={TTA_AP_DROPOUT}')
    print('=' * 70)

    rows = []
    seed_preds_baseline = []   # for ensemble comparison
    seed_preds_tta = []         # for ensemble comparison

    for seed in SEEDS:
        ckpt = CKPT_DIR / f'A_random__BestCombo_s{seed}.pt'
        if not ckpt.exists():
            print(f'[skip] missing checkpoint {ckpt}')
            continue
        model = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device).eval()

        # ── Baseline (no TTA) ────────────────────────────────
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            val_t = torch.from_numpy(set_val[te_A]).to(device)
            mask_t = torch.from_numpy(set_mask[te_A]).to(device)
            log_pi, mu, _ = model(idx_t, val_t, mask_t)
            k = log_pi.argmax(dim=-1)
            base_pred = mu[torch.arange(mu.size(0)), k].cpu().numpy()
        base_err = loc_err(base_pred, y_te)
        bs = loc_stats(base_err)

        # ── TTA ──────────────────────────────────────────────
        tta_pred, tta_unc, _, _, _, _ = tta_predict(model, set_idx[te_A],
                                                      set_val[te_A], set_mask[te_A],
                                                      device, seed=seed)
        tta_err = loc_err(tta_pred, y_te)
        ts = loc_stats(tta_err)

        print(f'seed {seed:2d}:  baseline median={bs["median_err_m"]:.3f}  '
              f'TTA median={ts["median_err_m"]:.3f}  '
              f'Δ={ts["median_err_m"]-bs["median_err_m"]:+.3f}')
        rows.append({'seed': seed, 'mode': 'baseline', **bs})
        rows.append({'seed': seed, 'mode': 'tta', **ts})
        seed_preds_baseline.append(base_pred)
        seed_preds_tta.append(tta_pred)

    # ── Ensemble baseline vs ensemble TTA ────────────────────
    print('\n' + '=' * 70)
    if len(seed_preds_baseline) >= 2:
        ens_base = np.mean(seed_preds_baseline, axis=0)
        ens_base_err = loc_err(ens_base, y_te)
        ens_base_s = loc_stats(ens_base_err)
        print(f'ENSEMBLE BASELINE (mean of {len(seed_preds_baseline)} seeds, no TTA):')
        print(f'  median={ens_base_s["median_err_m"]:.3f}  '
              f'mean={ens_base_s["mean_err_m"]:.3f}  '
              f'p90={ens_base_s["p90_err_m"]:.3f}')

        ens_tta = np.mean(seed_preds_tta, axis=0)
        ens_tta_err = loc_err(ens_tta, y_te)
        ens_tta_s = loc_stats(ens_tta_err)
        print(f'ENSEMBLE + TTA (mean of {len(seed_preds_tta)} seeds × {N_TTA} TTA):')
        print(f'  median={ens_tta_s["median_err_m"]:.3f}  '
              f'mean={ens_tta_s["mean_err_m"]:.3f}  '
              f'p90={ens_tta_s["p90_err_m"]:.3f}')

        delta = ens_tta_s['median_err_m'] - ens_base_s['median_err_m']
        print(f'  Δ vs baseline: {delta:+.3f} m '
              f'({100*delta/ens_base_s["median_err_m"]:+.1f}%)')

        # Save predictions
        np.savez(PRED_DIR / 'A_random__BigEnsembleTTA.npz',
                  pred=ens_tta, y_true=y_te, err=ens_tta_err, test_idx=te_A)

        # Append to metrics.csv
        csv = OUT_DIR / 'metrics.csv'
        new = pd.DataFrame([
            {'split': 'A_random', 'model': 'BigEnsembleNoTTA',
             **ens_base_s, 'p10_err_m': None, 'max_err_m': None, 'nll': None},
            {'split': 'A_random', 'model': 'BigEnsembleTTA',
             **ens_tta_s, 'p10_err_m': None, 'max_err_m': None, 'nll': None},
        ])
        # match columns
        if csv.exists():
            old = pd.read_csv(csv)
            old = old[~old['model'].isin(['BigEnsembleNoTTA', 'BigEnsembleTTA'])]
            # align columns
            for col in old.columns:
                if col not in new.columns:
                    new[col] = None
            full = pd.concat([old, new[old.columns]], ignore_index=True)
        else:
            full = new
        full.to_csv(csv, index=False)

    print('\nper-seed detail:')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
