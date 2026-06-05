"""Load the mega-ensemble winner: CascadeTuned (5 seeds) + EmbKNN.

Reproduces median 0.734 m on Split A — the current best result.

EmbKNN uses the CascadeTuned encoder as a fixed feature extractor + K=5
neighbour-weighted average over train labels. It has very different errors
from the Cascade decoder (retrieval-based, not heatmap), so mixing the two
via geometric median beats either family alone:

    Cascade-tuned alone  → 0.760
    EmbKNN alone         → 0.851
    Cascade-tuned + EmbKNN (geom-median over 6 prediction sets) → 0.734
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
SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fine_xy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    coarse_mask = data.build_free_mask(coarse_xy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    # load 5 CascadeTuned seeds
    preds = []                       # decoder predictions, per seed
    embs_tr = None                   # accumulated encoder embeddings on TRAIN
    embs_te = None                   # accumulated encoder embeddings on TEST
    for s in SEEDS:
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
            coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to(device)
        m.load_state_dict(torch.load(
            CKPT_DIR / f'A_random__CascadeTuned_s{s}.pt',
            map_location=device))
        m.eval()
        with torch.no_grad():
            it = torch.from_numpy(set_idx[te_A]).to(device)
            vt = torch.from_numpy(set_val[te_A]).to(device)
            mt = torch.from_numpy(set_mask[te_A]).to(device)
            preds.append(m.predict_xy(it, vt, mt))

            # encoder embedding for EmbKNN — batched
            def emb(idx_a, val_a, mask_a):
                out = []
                for st in range(0, idx_a.shape[0], 256):
                    en = min(st + 256, idx_a.shape[0])
                    o = m.encode(
                        torch.from_numpy(idx_a[st:en]).to(device),
                        torch.from_numpy(val_a[st:en]).to(device),
                        torch.from_numpy(mask_a[st:en]).to(device)).cpu().numpy()
                    out.append(o)
                return np.concatenate(out, axis=0)
            Z_tr = emb(set_idx[tr_A], set_val[tr_A], set_mask[tr_A])
            Z_te = emb(set_idx[te_A], set_val[te_A], set_mask[te_A])
        embs_tr = Z_tr if embs_tr is None else embs_tr + Z_tr
        embs_te = Z_te if embs_te is None else embs_te + Z_te

    n = len(SEEDS)
    Z_tr = embs_tr / n
    Z_te = embs_te / n
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-9)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-9)

    nn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(Z_tr)
    d, idx = nn.kneighbors(Z_te)
    w = 1.0 / (d + 1e-6); w = w / w.sum(axis=1, keepdims=True)
    pred_knn = (w[..., None] * y[tr_A][idx]).sum(axis=1)
    preds.append(pred_knn)

    pts = np.stack(preds, axis=0)
    pred = geom_median(pts)
    err = np.linalg.norm(pred - y[te_A], axis=1)

    print('\n=== Mega-ensemble: CascadeTuned 5-seed + EmbKNN (Split A test) ===')
    print(f'  median = {np.median(err):.3f} m   (expected ~0.734)')
    print(f'  mean   = {err.mean():.3f} m')
    print(f'  p90    = {np.percentile(err, 90):.3f} m')
    print(f'  within 0.3 m: {(err <= 0.3).mean() * 100:.1f}%')


if __name__ == '__main__':
    main()
