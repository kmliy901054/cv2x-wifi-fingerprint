"""Generate yet more EmbKNN variants to feed the mega-ensemble.
BaseK5 was best; add more BaseK options + an Ultra-encoder variant.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import data, models
from embknn_variants import get_embeddings

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PRED_DIR = HERE / 'outputs' / 'predictions'


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    configs = [
        ('Cascade',           3, 'EmbKNN_BaseK3'),
        ('Cascade',           7, 'EmbKNN_BaseK7'),
        ('Cascade',          10, 'EmbKNN_BaseK10'),
        ('Cascade',          15, 'EmbKNN_BaseK15'),
        ('CascadeUltra',      5, 'EmbKNN_UltraK5'),
        ('CascadeUltra',      7, 'EmbKNN_UltraK7'),
    ]
    for model_name, K, save_name in configs:
        print(f'\n=== {save_name} (encoder={model_name}, K={K}) ===')
        Z_tr, Z_te = get_embeddings(model_name,
                                       set_idx[tr_A], set_val[tr_A], set_mask[tr_A],
                                       set_idx[te_A], set_val[te_A], set_mask[te_A],
                                       bssids, fxy, fmask, cxy, cmask, f2c)
        if Z_tr is None:
            print(f'  no checkpoints, skip'); continue
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_tr)
        d, idx = nn.kneighbors(Z_te)
        w = 1.0 / (d + 1e-6); w = w / w.sum(axis=1, keepdims=True)
        pred = (w[..., None] * y[tr_A][idx]).sum(axis=1)
        err = np.linalg.norm(pred - y[te_A], axis=1)
        print(f'  median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}')
        np.savez(PRED_DIR / f'A_random__{save_name}.npz',
                  pred=pred, y_true=y[te_A], err=err, test_idx=te_A)


if __name__ == '__main__':
    main()
