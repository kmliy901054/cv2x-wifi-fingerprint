"""Generate multiple EmbKNN predictions using different encoders and K values
to add more diverse retrieval-based members to the mega-ensemble.

EmbKNN variants saved (each as a separate prediction set):
  EmbKNN_BaseK5    — baseline Cascade encoder, K=5
  EmbKNN_TunedK5   — CascadeTuned encoder, K=5 (already exists as EmbKNN)
  EmbKNN_AggrK5    — CascadeAggressive encoder, K=5
  EmbKNN_TunedK10  — CascadeTuned encoder, K=10
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import data, models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / 'outputs' / 'checkpoints'
PRED_DIR = HERE / 'outputs' / 'predictions'
SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)


def loc_err(yp, yt): return np.linalg.norm(yp - yt, axis=1)


def embed_with(model_name, set_idx, set_val, set_mask, bssids, fxy, fmask,
                cxy, cmask, f2c):
    """Average embeddings across the 5 seeds of `model_name`."""
    Z_tr = None; Z_te = None
    n_loaded = 0
    for s in SEEDS:
        ck = CKPT_DIR / f'A_random__{model_name}_s{s}.pt'
        if not ck.exists(): continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to('cpu')
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            def emb(idx_a, val_a, mask_a):
                out = []
                for st in range(0, idx_a.shape[0], 256):
                    en = min(st + 256, idx_a.shape[0])
                    o = m.encode(torch.from_numpy(idx_a[st:en]),
                                  torch.from_numpy(val_a[st:en]),
                                  torch.from_numpy(mask_a[st:en])).numpy()
                    out.append(o)
                return np.concatenate(out, axis=0)
            zt = emb(set_idx, set_val, set_mask)
        Z_tr = zt if Z_tr is None else Z_tr  # placeholder; computed per call
        Z_te = zt
        n_loaded += 1
    return n_loaded


def get_embeddings(model_name, idx_tr, val_tr, mask_tr, idx_te, val_te, mask_te,
                     bssids, fxy, fmask, cxy, cmask, f2c):
    Z_tr_acc = None; Z_te_acc = None; n = 0
    for s in SEEDS:
        ck = CKPT_DIR / f'A_random__{model_name}_s{s}.pt'
        if not ck.exists():
            print(f'  [skip] {ck.name}')
            continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to('cpu')
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            def emb(idx_a, val_a, mask_a):
                out = []
                for st in range(0, idx_a.shape[0], 256):
                    en = min(st + 256, idx_a.shape[0])
                    o = m.encode(torch.from_numpy(idx_a[st:en]),
                                  torch.from_numpy(val_a[st:en]),
                                  torch.from_numpy(mask_a[st:en])).numpy()
                    out.append(o)
                return np.concatenate(out, axis=0)
            Z_tr = emb(idx_tr, val_tr, mask_tr)
            Z_te = emb(idx_te, val_te, mask_te)
        Z_tr_acc = Z_tr if Z_tr_acc is None else Z_tr_acc + Z_tr
        Z_te_acc = Z_te if Z_te_acc is None else Z_te_acc + Z_te
        n += 1
    if n == 0: return None, None
    Z_tr = Z_tr_acc / n; Z_te = Z_te_acc / n
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-9)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-9)
    return Z_tr, Z_te


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    configs = [
        ('Cascade',           5, 'EmbKNN_BaseK5'),
        ('CascadeTuned',      3, 'EmbKNN_TunedK3'),
        ('CascadeTuned',      7, 'EmbKNN_TunedK7'),
        ('CascadeTuned',     10, 'EmbKNN_TunedK10'),
        ('CascadeAggressive', 5, 'EmbKNN_AggrK5'),
    ]

    for model_name, K, save_name in configs:
        print(f'\n=== {save_name} (encoder={model_name}, K={K}) ===')
        Z_tr, Z_te = get_embeddings(model_name,
                                       set_idx[tr_A], set_val[tr_A], set_mask[tr_A],
                                       set_idx[te_A], set_val[te_A], set_mask[te_A],
                                       bssids, fxy, fmask, cxy, cmask, f2c)
        if Z_tr is None:
            print(f'  no checkpoints found, skipping')
            continue
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_tr)
        d, idx = nn.kneighbors(Z_te)
        w = 1.0 / (d + 1e-6); w = w / w.sum(axis=1, keepdims=True)
        pred = (w[..., None] * y[tr_A][idx]).sum(axis=1)
        err = loc_err(pred, y[te_A])
        print(f'  median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}')
        np.savez(PRED_DIR / f'A_random__{save_name}.npz',
                  pred=pred, y_true=y[te_A], err=err, test_idx=te_A)


if __name__ == '__main__':
    main()
