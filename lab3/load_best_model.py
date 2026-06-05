"""Load the pushed winning model and reproduce the Split A result.

By default loads **Cascade-tuned** (median 0.760 m, the current winner).
Pass --variant baseline to load the original Cascade (median 0.793 m).

Both variants' 5-seed checkpoints are committed to the repo (~36 MB total).
All other experiment checkpoints are gitignored; rerun the corresponding
train_*.py to regenerate them.

Usage:
    python load_best_model.py                    # tuned (0.760 m, default)
    python load_best_model.py --variant baseline # baseline Cascade (0.793 m)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import data
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CKPT_DIR = Path(__file__).parent / 'outputs' / 'checkpoints'
SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
VARIANTS = {
    'tuned':    {'ckpt': 'A_random__CascadeTuned_s{s}.pt', 'expected': 0.760},
    'baseline': {'ckpt': 'A_random__Cascade_s{s}.pt',      'expected': 0.793},
}


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d
        w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=list(VARIANTS), default='tuned',
                    help='which 5-seed ensemble to load (default: tuned)')
    args = ap.parse_args()
    variant = VARIANTS[args.variant]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}    variant: {args.variant}')

    # data + Split A test set
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    _, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    # grids needed to instantiate the Cascade model
    fine_xy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    coarse_mask = data.build_free_mask(coarse_xy)
    fine_to_coarse = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    preds = []
    for s in SEEDS:
        ckpt = CKPT_DIR / variant['ckpt'].format(s=s)
        if not ckpt.exists():
            print(f'[skip] missing {ckpt.name}')
            continue
        model = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
            coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
            fine_to_coarse=fine_to_coarse, **CFG).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            v_t = torch.from_numpy(set_val[te_A]).to(device)
            m_t = torch.from_numpy(set_mask[te_A]).to(device)
            preds.append(model.predict_xy(idx_t, v_t, m_t))
        print(f'  loaded seed {s}')

    pts = np.stack(preds, axis=0)
    pred = geom_median(pts)
    err = np.linalg.norm(pred - y[te_A], axis=1)
    print(f'\n=== Cascade ({args.variant}) 5-seed geom-median (Split A test) ===')
    print(f'  median = {np.median(err):.3f} m   '
          f'(expected ~{variant["expected"]:.3f})')
    print(f'  mean   = {err.mean():.3f} m')
    print(f'  p90    = {np.percentile(err, 90):.3f} m')
    print(f'  within 0.3 m (AMCL noise floor): {(err <= 0.3).mean() * 100:.1f}%')


if __name__ == '__main__':
    main()
