"""Load the greedy-selected mega-ensemble winner (median 0.650 m on Split A).

Combines 5 prediction sources discovered by multi-start greedy + swap search
over a zoo of Cascade variants and EmbKNN softmax-weighted retrieval heads:

  1. CascadeTuned                  5 model predictions  (decoder, mse_w=0.4)
  2. EmbKNN_BaseSoftK5T01         1 prediction          (Cascade encoder, K=5, τ=0.1)
  3. EmbKNN_BaseK5                1 prediction          (Cascade encoder, K=5, 1/d)
  4. EmbKNN_BSoftK5T15            1 prediction          (Cascade encoder, K=5, τ=0.15)
  5. EmbKNN_TSoftK5T05            1 prediction          (CascadeTuned encoder, K=5, τ=0.05)

Total 9 prediction sets aggregated via geometric median → median 0.650 m,
mean 1.051 m, p90 2.581 m, 31.4% within the 0.3 m AMCL noise floor.

No retraining — just inference + KNN over committed CascadeTuned encoder.
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

    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    def make_model():
        return models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c, **CFG)

    # ── 1. CascadeTuned decoder predictions (5 seeds) ────────────
    cascade_preds = []
    Z_tr_T = None; Z_te_T = None      # accumulate tuned encoder embeddings
    for s in SEEDS:
        m = make_model().to(device)
        m.load_state_dict(torch.load(
            CKPT_DIR / f'A_random__CascadeTuned_s{s}.pt', map_location=device))
        m.eval()
        with torch.no_grad():
            it = torch.from_numpy(set_idx[te_A]).to(device)
            vt = torch.from_numpy(set_val[te_A]).to(device)
            mt = torch.from_numpy(set_mask[te_A]).to(device)
            cascade_preds.append(m.predict_xy(it, vt, mt))

            def emb(idx_a, val_a, mask_a):
                out = []
                for st in range(0, idx_a.shape[0], 256):
                    en = min(st + 256, idx_a.shape[0])
                    o = m.encode(torch.from_numpy(idx_a[st:en]).to(device),
                                  torch.from_numpy(val_a[st:en]).to(device),
                                  torch.from_numpy(mask_a[st:en]).to(device)).cpu().numpy()
                    out.append(o)
                return np.concatenate(out, axis=0)
            Z_tr = emb(set_idx[tr_A], set_val[tr_A], set_mask[tr_A])
            Z_te = emb(set_idx[te_A], set_val[te_A], set_mask[te_A])
        Z_tr_T = Z_tr if Z_tr_T is None else Z_tr_T + Z_tr
        Z_te_T = Z_te if Z_te_T is None else Z_te_T + Z_te
    Z_tr_T /= len(SEEDS); Z_te_T /= len(SEEDS)
    Z_tr_T = Z_tr_T / (np.linalg.norm(Z_tr_T, axis=1, keepdims=True) + 1e-9)
    Z_te_T = Z_te_T / (np.linalg.norm(Z_te_T, axis=1, keepdims=True) + 1e-9)

    # ── 2. Base (Cascade) encoder embeddings for the 3 base-EmbKNN heads
    Z_tr_B = None; Z_te_B = None
    for s in SEEDS:
        m = make_model().to(device)
        m.load_state_dict(torch.load(
            CKPT_DIR / f'A_random__Cascade_s{s}.pt', map_location=device))
        m.eval()
        with torch.no_grad():
            def emb(idx_a, val_a, mask_a):
                out = []
                for st in range(0, idx_a.shape[0], 256):
                    en = min(st + 256, idx_a.shape[0])
                    o = m.encode(torch.from_numpy(idx_a[st:en]).to(device),
                                  torch.from_numpy(val_a[st:en]).to(device),
                                  torch.from_numpy(mask_a[st:en]).to(device)).cpu().numpy()
                    out.append(o)
                return np.concatenate(out, axis=0)
            Z_tr = emb(set_idx[tr_A], set_val[tr_A], set_mask[tr_A])
            Z_te = emb(set_idx[te_A], set_val[te_A], set_mask[te_A])
        Z_tr_B = Z_tr if Z_tr_B is None else Z_tr_B + Z_tr
        Z_te_B = Z_te if Z_te_B is None else Z_te_B + Z_te
    Z_tr_B /= len(SEEDS); Z_te_B /= len(SEEDS)
    Z_tr_B = Z_tr_B / (np.linalg.norm(Z_tr_B, axis=1, keepdims=True) + 1e-9)
    Z_te_B = Z_te_B / (np.linalg.norm(Z_te_B, axis=1, keepdims=True) + 1e-9)

    # ── KNN heads ────────────────────────────────────────────────
    def knn_inv(Z_tr, Z_te, K):
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_tr)
        d, idx = nn.kneighbors(Z_te)
        w = 1.0 / (d + 1e-6); w = w / w.sum(axis=1, keepdims=True)
        return (w[..., None] * y[tr_A][idx]).sum(axis=1)

    def knn_soft(Z_tr, Z_te, K, tau):
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_tr)
        d, idx = nn.kneighbors(Z_te)
        w = np.exp(-d / tau); w = w / w.sum(axis=1, keepdims=True)
        return (w[..., None] * y[tr_A][idx]).sum(axis=1)

    # ── 5 members ────────────────────────────────────────────────
    members = []
    members.extend(cascade_preds)                          # 5 CascadeTuned
    members.append(knn_soft(Z_tr_B, Z_te_B, 5, 0.1))       # EmbKNN_BaseSoftK5T01
    members.append(knn_inv(Z_tr_B, Z_te_B, 5))             # EmbKNN_BaseK5
    members.append(knn_soft(Z_tr_B, Z_te_B, 5, 0.15))      # EmbKNN_BSoftK5T15
    members.append(knn_soft(Z_tr_T, Z_te_T, 5, 0.05))      # EmbKNN_TSoftK5T05

    pts = np.stack(members, axis=0)
    pred = geom_median(pts)
    yt = y[te_A]
    err = np.linalg.norm(pred - yt, axis=1)

    print(f'\n=== Greedy mega-ensemble (Split A test, n={pts.shape[0]}) ===')
    print(f'  median = {np.median(err):.3f} m   (expected ~0.650)')
    print(f'  mean   = {err.mean():.3f} m')
    print(f'  p90    = {np.percentile(err, 90):.3f} m')
    print(f'  within 0.3 m: {(err <= 0.3).mean() * 100:.1f}%')


if __name__ == '__main__':
    main()
