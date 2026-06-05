"""Try alternative aggregation methods on the current top mega-ensemble combo.

geom-median already used; try: trimmed-mean, inverse-error weighted,
medoid, and per-axis median.
"""
import sys
from pathlib import Path
import numpy as np
import torch
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


def loc_err(p, y): return np.linalg.norm(p - y, axis=1)


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d; w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps: break
        m = new
    return m


def trimmed_mean(pts, trim_frac=0.2):
    """For each test sample, drop top-frac and bottom-frac by distance to mean."""
    N = pts.shape[0]
    k_trim = int(N * trim_frac)
    if k_trim == 0:
        return pts.mean(axis=0)
    centroid = pts.mean(axis=0)                          # (T, 2)
    d = np.linalg.norm(pts - centroid[None], axis=-1)    # (N, T)
    # rank along ensemble axis
    keep_mask = np.ones_like(d, dtype=bool)
    sort_idx = np.argsort(d, axis=0)
    for t in range(d.shape[1]):
        # drop the k_trim FARTHEST predictions for this test sample
        keep_mask[sort_idx[-k_trim:, t], t] = False
    pts_masked = pts.copy()
    pts_masked[~keep_mask] = 0
    counts = keep_mask.sum(axis=0)                       # (T,)
    s = pts_masked.sum(axis=0)
    return s / counts[:, None]


def medoid(pts):
    """For each test sample, pick the prediction that's closest to all others."""
    N, T, _ = pts.shape
    d = np.zeros((N, T))
    for i in range(N):
        d[i] = np.linalg.norm(pts - pts[i:i+1], axis=-1).sum(axis=0)
    best = np.argmin(d, axis=0)                           # (T,)
    return np.take_along_axis(pts, best[None, :, None],
                                axis=0).squeeze(0)


def per_axis_median(pts):
    return np.median(pts, axis=0)


def load_pred_or_npz(name, **k):
    """Load saved npz prediction OR a Cascade-variant by loading 5 seeds."""
    f = PRED_DIR / f'A_random__{name}.npz'
    if f.exists():
        return np.load(f)['pred'][None, ...]              # (1, T, 2)
    # else assume it's a Cascade variant
    return load_cascade_variant(name, **k)


def load_cascade_variant(name, bssids, fxy, fmask, cxy, cmask, f2c,
                            set_idx, set_val, set_mask, te_A):
    device = torch.device('cpu')
    preds = []
    for s in SEEDS:
        ck = CKPT_DIR / f'A_random__{name}_s{s}.pt'
        if not ck.exists(): continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to(device)
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            it = torch.from_numpy(set_idx[te_A])
            vt = torch.from_numpy(set_val[te_A])
            mt = torch.from_numpy(set_mask[te_A])
            preds.append(m.predict_xy(it, vt, mt))
    return np.stack(preds, axis=0)


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)
    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    parts = ['CascadeAggressive', 'EmbKNN_BaseK7', 'EmbKNN_BaseK10',
              'EmbKNN_TunedK7']
    stacked = []
    for name in parts:
        p = load_pred_or_npz(name, bssids=bssids, fxy=fxy, fmask=fmask,
                              cxy=cxy, cmask=cmask, f2c=f2c,
                              set_idx=set_idx, set_val=set_val,
                              set_mask=set_mask, te_A=te_A)
        stacked.append(p)
        print(f'  {name}: {p.shape[0]} preds')
    stacked = np.concatenate(stacked, axis=0)                # (N_models, T, 2)
    yt = y[te_A]
    print(f'\nstacked: {stacked.shape}')

    for tag, p in [
        ('mean',           stacked.mean(axis=0)),
        ('geom-median',    geom_median(stacked)),
        ('per-axis-med',   per_axis_median(stacked)),
        ('medoid',         medoid(stacked)),
        ('trim20-mean',    trimmed_mean(stacked, trim_frac=0.20)),
        ('trim40-mean',    trimmed_mean(stacked, trim_frac=0.40)),
    ]:
        e = loc_err(p, yt)
        print(f'  {tag:<16}  median={np.median(e):.3f}  mean={e.mean():.3f}  '
              f'p90={np.percentile(e, 90):.3f}')


if __name__ == '__main__':
    main()
