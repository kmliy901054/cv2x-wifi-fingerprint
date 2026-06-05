"""Mega-ensemble evaluation.

Loads up to 15 Cascade-style models from 3 configurations and tries every
ensemble combination, both arithmetic-mean and geometric-median, to find the
best Split A median.

Variants:
  baseline:   Cascade (mse_w=0.2, fine_sigma=0.4)        5 seeds
  tuned:      Cascade-tuned (mse_w=0.4, fine_sigma=0.3)  5 seeds
  aggressive: Cascade-aggressive (mse_w=0.55, σ=0.25)    5 seeds
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import data
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / 'outputs' / 'checkpoints'
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ['Cascade', 'CascadeTuned', 'CascadeAggressive']
# extra single-model predictions saved as npz (no checkpoint reload needed)
EXTRA_NPZ = ['EmbKNN', 'GB']


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d; w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


def stats(err):
    return (float(np.median(err)), float(err.mean()),
            float(np.percentile(err, 90)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    _, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fine_xy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    coarse_mask = data.build_free_mask(coarse_xy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    # collect predictions, indexed by variant name
    variant_preds = {}
    for v in VARIANTS:
        preds = []
        for s in SEEDS:
            ck = CKPT_DIR / f'A_random__{v}_s{s}.pt'
            if not ck.exists():
                print(f'[skip] missing {ck.name}')
                continue
            m = models.SetTransformerHeatmapCascade(
                num_bssids=len(bssids),
                fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
                coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
                fine_to_coarse=f2c, **CFG).to(device)
            m.load_state_dict(torch.load(ck, map_location=device))
            m.eval()
            with torch.no_grad():
                it = torch.from_numpy(set_idx[te_A]).to(device)
                vt = torch.from_numpy(set_val[te_A]).to(device)
                mt = torch.from_numpy(set_mask[te_A]).to(device)
                preds.append(m.predict_xy(it, vt, mt))
        variant_preds[v] = np.stack(preds, axis=0) if preds else None
        if preds:
            print(f'[ok] {v}: loaded {len(preds)} seeds')

    # extra single-model preds from npz
    PRED_DIR = HERE / 'outputs' / 'predictions'
    for name in EXTRA_NPZ:
        f = PRED_DIR / f'A_random__{name}.npz'
        if f.exists():
            d = np.load(f)
            variant_preds[name] = d['pred'][None, ...]            # (1, N, 2)
            print(f'[ok] {name}: loaded saved npz')

    yt = y[te_A]

    def ev(stacked, tag):
        m_pred = stacked.mean(axis=0)
        gm_pred = geom_median(stacked)
        med_m, mean_m, p90_m = stats(np.linalg.norm(m_pred - yt, axis=1))
        med_g, mean_g, p90_g = stats(np.linalg.norm(gm_pred - yt, axis=1))
        return (med_m, mean_m, p90_m, med_g, mean_g, p90_g)

    print(f'\n{"combination":<55}{"size":>5}  {"mean median":>11}  {"geom median":>11}')
    print('-' * 92)
    results = []
    # per-variant 5-seed
    for v in VARIANTS:
        if variant_preds[v] is None: continue
        med_m, _, _, med_g, _, _ = ev(variant_preds[v], v)
        print(f'  {v:<53}{len(variant_preds[v]):>5}  {med_m:>11.3f}  {med_g:>11.3f}')
        results.append((f'{v}', med_g))
    # show extras as their own row
    for name in EXTRA_NPZ:
        if name in variant_preds:
            stacked = variant_preds[name]
            med_m, _, _, med_g, _, _ = ev(stacked, name)
            print(f'  {name:<53}{len(stacked):>5}  {med_m:>11.3f}  {med_g:>11.3f}')
            results.append((name, med_g))

    # all combos of variants (Cascade family only first, then add extras)
    all_avail = [v for v in VARIANTS + EXTRA_NPZ if v in variant_preds]
    for r in range(2, len(all_avail) + 1):
        for combo in combinations(all_avail, r):
            stacked = np.concatenate([variant_preds[v] for v in combo], axis=0)
            med_m, _, _, med_g, _, _ = ev(stacked, '+'.join(combo))
            print(f'  {"+".join(combo):<53}{len(stacked):>5}  {med_m:>11.3f}  {med_g:>11.3f}')
            results.append(('+'.join(combo), med_g))

    print('-' * 92)
    best = sorted(results, key=lambda x: x[1])[:5]
    print('\nTop-5 by geom-median:')
    for name, med in best:
        print(f'  {name:<55}{med:.3f}')

    # save best combo prediction
    best_name = best[0][0]
    if '+' in best_name:
        combo = best_name.split('+')
    else:
        combo = [best_name]
    stacked = np.concatenate([variant_preds[v] for v in combo], axis=0)
    pred = geom_median(stacked)
    err = np.linalg.norm(pred - yt, axis=1)
    print(f'\n=== Best ensemble: {best_name} (n={len(stacked)}) ===')
    print(f'  median = {np.median(err):.3f}')
    print(f'  mean   = {err.mean():.3f}')
    print(f'  p90    = {np.percentile(err, 90):.3f}')
    print(f'  within 0.3 m: {(err <= 0.3).mean() * 100:.1f}%')
    np.savez(HERE / 'outputs' / 'predictions' / 'A_random__MegaEnsemble.npz',
              pred=pred, y_true=yt, err=err, test_idx=te_A,
              combo=np.array(best_name))


if __name__ == '__main__':
    main()
