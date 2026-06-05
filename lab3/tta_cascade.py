"""Test-Time Augmentation on Cascade winning ensemble.

For each test scan, generate K augmented copies (RSSI jitter + AP dropout)
and average the gated fine probabilities across all (K * 5 seeds) outputs
before taking the expected (x, y). Probability-space averaging is more
principled than point averaging because the model output is a distribution.

Reuses the committed Cascade weights — no retraining.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
import models

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
        w = 1.0 / d
        w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


def augment_batch(idx, val, mask, jitter_dbm, ap_dropout, rng):
    """Apply RSSI jitter + AP dropout to a batch of inputs."""
    val = val.clone()
    mask = mask.clone()
    if jitter_dbm > 0:
        noise = (torch.rand(val.shape, generator=rng, device=val.device,
                             dtype=val.dtype) * 2 - 1) * (jitter_dbm / 20.0)
        val = val + noise * mask                            # only on real APs
    if ap_dropout > 0:
        keep = (torch.rand(mask.shape, generator=rng, device=mask.device,
                            dtype=mask.dtype) > ap_dropout).float()
        mask = mask * keep
    return idx, val, mask


@torch.no_grad()
def tta_predict(model_list, set_idx, set_val, set_mask,
                  device, K=20, jitter=2.0, ap_drop=0.05):
    """Return (N, 2) ensemble-TTA predictions.

    For each model and each augmentation step, compute the gated fine
    probability. Average across all 5*K probabilities, then expected (x, y).
    """
    idx = torch.from_numpy(set_idx).to(device)
    val0 = torch.from_numpy(set_val).to(device)
    mask0 = torch.from_numpy(set_mask).to(device)
    N = idx.size(0)
    # accumulate probability sum on GPU
    prob_sum = None
    rng = torch.Generator(device=device).manual_seed(0)
    M = len(model_list)
    total = K * M
    for k in range(K):
        if k == 0:
            it, vt, mt = idx, val0, mask0                   # raw, no aug
        else:
            it, vt, mt = augment_batch(idx, val0, mask0, jitter, ap_drop, rng)
        for m in model_list:
            f_l, c_l = m(it, vt, mt)
            p = m._gated_fine_prob(f_l, c_l)                # (N, Gf)
            prob_sum = p if prob_sum is None else prob_sum + p
    prob_avg = prob_sum / float(total)
    fine_xy = model_list[0].fine_xy
    return (prob_avg @ fine_xy).cpu().numpy()


@torch.no_grad()
def baseline_predict(model_list, set_idx, set_val, set_mask, device):
    """No-TTA, per-seed predictions stacked → geom-median ensemble."""
    idx = torch.from_numpy(set_idx).to(device)
    val = torch.from_numpy(set_val).to(device)
    mask = torch.from_numpy(set_mask).to(device)
    preds = []
    for m in model_list:
        preds.append(m.predict_xy(idx, val, mask))
    pts = np.stack(preds, axis=0)
    return geom_median(pts)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

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

    model_list = []
    for s in SEEDS:
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
            coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to(device)
        m.load_state_dict(torch.load(CKPT_DIR / f'A_random__Cascade_s{s}.pt',
                                       map_location=device))
        m.eval()
        model_list.append(m)
    print(f'loaded {len(model_list)} models', flush=True)

    # Baseline (no TTA)
    print('\n=== Baseline (5-seed geom-median, no TTA) ===', flush=True)
    pred = baseline_predict(model_list, set_idx[te_A], set_val[te_A],
                              set_mask[te_A], device)
    err = np.linalg.norm(pred - y[te_A], axis=1)
    print(f'  median={np.median(err):.3f}  mean={err.mean():.3f}  '
          f'p90={np.percentile(err, 90):.3f}', flush=True)

    # TTA sweep over (K, jitter, ap_drop)
    configs = [
        (10, 1.0, 0.00),
        (10, 2.0, 0.00),
        (20, 2.0, 0.00),
        (20, 2.0, 0.05),
        (20, 3.0, 0.05),
        (40, 2.0, 0.05),
    ]
    for K, jit, drop in configs:
        t0 = time.time()
        pred = tta_predict(model_list, set_idx[te_A], set_val[te_A],
                             set_mask[te_A], device,
                             K=K, jitter=jit, ap_drop=drop)
        err = np.linalg.norm(pred - y[te_A], axis=1)
        dt = time.time() - t0
        print(f'  K={K:2d} jit={jit:.1f}dB drop={drop:.2f}  '
              f'median={np.median(err):.3f}  '
              f'mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}   ({dt:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
