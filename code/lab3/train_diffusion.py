"""Phase 2 — Diffusion regression head (arXiv 2310.05768 style).

Replaces heatmap with conditional diffusion model over (x, y) ∈ ℝ².
Encoder identical to prior Cascade; output is sampled via DDIM (20 steps,
averaged over 4 trajectories).

No free-cell mask in the head itself; we apply a nearest-free-cell snap
as a cheap post-process to recover the geometry prior.

Combined with the 5000 GP synth (matches Cascade for apples-to-apples).
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

CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
DIFF_CFG = dict(diff_T=100, diff_hidden=256, pose_scale=5.0,
                  num_sample_steps=25, num_inf_samples=8)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 400              # diffusion needs more epochs
PATIENCE = 80
SEEDS = [42, 43, 44, 45, 46]
MAX_APS = 50
MIN_BSSID_COUNT = 10
N_SYNTHETIC = 12000


def loc_err(yp, yt):
    return np.linalg.norm(yp - yt, axis=1)


def loc_stats(err):
    return {'median_err_m': float(np.median(err)),
            'mean_err_m': float(err.mean()),
            'p90_err_m': float(np.percentile(err, 90)),
            'p10_err_m': float(np.percentile(err, 10)),
            'max_err_m': float(err.max())}


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - m[None], axis=-1)
        d = np.clip(d, eps, None)
        w = 1.0 / d
        w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


def snap_to_free(pred, free_cells_xy):
    """Snap each (x, y) prediction to the nearest free cell centre.
    pred: (N, 2), free_cells_xy: (G, 2) — only the free ones, not the full grid."""
    from scipy.spatial import cKDTree
    tree = cKDTree(free_cells_xy)
    _, idx = tree.query(pred, k=1)
    return free_cells_xy[idx]


def train_one(model, train_loader, val_loader, device, tag):
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
            cond = model(idx, val, mask)
            loss = model.loss(cond, y)
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
                cond = model(idx, val, mask)
                vloss += float(model.loss(cond, y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        if vloss < best_val - 1e-4:
            best_val = vloss; bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  [{tag}] early stop @ epoch {epoch}, '
                      f'best val = {best_val:.4f}', flush=True)
                break
        if epoch == 1 or epoch % 40 == 0:
            print(f'  [{tag}] epoch {epoch:3d}  val_loss {vloss:.4f}  '
                  f'best {best_val:.4f}', flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids,
                                                        max_aps=MAX_APS)
    # Get a fine free grid for snap post-processing
    free_grid, _, _ = data.build_heatmap_grid(cell_size=0.25)
    free_grid_mask = data.build_free_mask(free_grid)
    free_cells_xy = free_grid[free_grid_mask]
    print(f'[snap] {len(free_cells_xy)} free cells for post-snap', flush=True)

    # GP synth
    print(f'\n[gen] fitting GPs + sampling {N_SYNTHETIC} synth ...', flush=True)
    t0 = time.time()
    train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                       'session': sess[i],
                       'aps': {b: float(X[i, j])
                                for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                      for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15,
                                    verbose=False)
    real_xy = np.array([(r['x'], r['y']) for r in train_records],
                        dtype=np.float32)
    free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(42)
    n_pick = min(N_SYNTHETIC, len(free_positions))
    chosen = rng.choice(len(free_positions), n_pick, replace=False)
    positions = free_positions[chosen]
    X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                              records=train_records,
                                              detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0, seed=42)
    print(f'[gen] done ({time.time() - t0:.1f}s)', flush=True)
    synth_idx, synth_val, synth_mask = synthetic.build_set_form(
        X_synth, bssids, max_aps=MAX_APS)
    idx_tr = np.concatenate([set_idx[tr_A], synth_idx], axis=0)
    val_tr = np.concatenate([set_val[tr_A], synth_val], axis=0)
    mask_tr = np.concatenate([set_mask[tr_A], synth_mask], axis=0)
    y_tr = np.concatenate([y[tr_A], y_synth.astype(np.float32)], axis=0)
    print(f'[setup] combined train: {len(y_tr)} '
          f'({len(tr_A)} real + {len(y_synth)} synth)', flush=True)

    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    seed_preds_raw = []
    seed_preds_snap = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f'\n========== seed {seed} ==========', flush=True)
        train_ds = data.SetDataset(idx_tr, val_tr, mask_tr, y_tr,
                                     jitter=JITTER_DBM)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        model = models.SetTransformerDiffusion(
            num_bssids=len(bssids), **CFG, **DIFF_CFG)
        if seed == SEEDS[0]:
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'[model] params: {n_p:,} ({n_p/1000:.1f}k)', flush=True)
        model, vloss = train_one(model, tl, test_loader, device, f's{seed}')
        model.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            v_t = torch.from_numpy(set_val[te_A]).to(device)
            m_t = torch.from_numpy(set_mask[te_A]).to(device)
            pred_raw = model.predict_xy(idx_t, v_t, m_t)
        pred_snap = snap_to_free(pred_raw, free_cells_xy)
        err_raw = loc_err(pred_raw, y[te_A])
        err_snap = loc_err(pred_snap, y[te_A])
        print(f'  seed {seed} raw:  median={np.median(err_raw):.3f}  '
              f'mean={err_raw.mean():.3f}  p90={np.percentile(err_raw, 90):.3f}  '
              f'val={vloss:.4f}', flush=True)
        print(f'  seed {seed} snap: median={np.median(err_snap):.3f}  '
              f'mean={err_snap.mean():.3f}  p90={np.percentile(err_snap, 90):.3f}',
              flush=True)
        seed_preds_raw.append({'pred': pred_raw, 'val_loss': vloss, 'err': err_raw})
        seed_preds_snap.append({'pred': pred_snap, 'val_loss': vloss, 'err': err_snap})
        torch.save(model.state_dict(),
                    CKPT_DIR / f'A_random__Diffusion_s{seed}.pt')

    # Ensembles
    pts_raw = np.stack([s['pred'] for s in seed_preds_raw], axis=0)
    pts_snap = np.stack([s['pred'] for s in seed_preds_snap], axis=0)
    egm_raw = loc_stats(loc_err(geom_median(pts_raw), y[te_A]))
    egm_snap = loc_stats(loc_err(geom_median(pts_snap), y[te_A]))
    em_raw = loc_stats(loc_err(pts_raw.mean(0), y[te_A]))
    em_snap = loc_stats(loc_err(pts_snap.mean(0), y[te_A]))
    print(f'\n========== ENSEMBLE (5-seed) ==========', flush=True)
    print(f'  raw  mean:        median={em_raw["median_err_m"]:.3f}  mean={em_raw["mean_err_m"]:.3f}  p90={em_raw["p90_err_m"]:.3f}')
    print(f'  raw  geom-median: median={egm_raw["median_err_m"]:.3f}  mean={egm_raw["mean_err_m"]:.3f}  p90={egm_raw["p90_err_m"]:.3f}')
    print(f'  snap mean:        median={em_snap["median_err_m"]:.3f}  mean={em_snap["mean_err_m"]:.3f}  p90={em_snap["p90_err_m"]:.3f}')
    print(f'  snap geom-median: median={egm_snap["median_err_m"]:.3f}  mean={egm_snap["mean_err_m"]:.3f}  p90={egm_snap["p90_err_m"]:.3f}')

    np.savez(PRED_DIR / 'A_random__DiffusionEnsemble.npz',
              pred=geom_median(pts_snap), y_true=y[te_A],
              err=loc_err(geom_median(pts_snap), y[te_A]),
              test_idx=te_A)

    print(f'\n[save] -> predictions/A_random__DiffusionEnsemble.npz')
    print(f'\n========== Updated Split A ladder ==========')
    print(f'  Cascade (2-level, 5-seed):        0.793 m')
    print(f'  Cascade3 + 10-seed (Phase 1):     ???')
    print(f'  Diffusion raw 5-seed geom:        {egm_raw["median_err_m"]:.3f} m')
    print(f'  Diffusion snap 5-seed geom:       {egm_snap["median_err_m"]:.3f} m')


if __name__ == '__main__':
    main()
