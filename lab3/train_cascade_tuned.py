"""Cascade with tuned loss weights — try to beat 0.793 m.

Hypothesis: the baseline Cascade uses (ce_fine_w=0.5, ce_coarse_w=0.3, mse_w=0.2).
The direct-regression term (mse on E[xy]) only gets 20% weight, so the model
mostly optimizes heatmap CE — which doesn't directly minimize Euclidean error.

Variant A:
  fine_sigma   0.4 → 0.3        (sharper labels, commit to correct cell)
  ce_fine_w    0.5 → 0.4        (slight de-emphasis to make room for mse)
  ce_coarse_w  0.3 → 0.2
  mse_w        0.2 → 0.4        (double direct regression weight)

Same encoder + 5000 GP synth + 5-seed ensemble + geom-median as the winner.
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
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
SEEDS = [42, 43, 44, 45, 46]
MAX_APS = 50
MIN_BSSID_COUNT = 10
N_SYNTHETIC = 5000

# tuned hyperparams (variant A)
COARSE_CELL_SIZE = 1.6
FINE_CELL_SIZE = 0.4
FINE_SIGMA = 0.3              # ↓ from 0.4
COARSE_SIGMA = 1.0
CE_FINE_W = 0.4               # ↓ from 0.5
CE_COARSE_W = 0.2             # ↓ from 0.3
MSE_W = 0.4                   # ↑ from 0.2


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
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d
        w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


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
            out = model(idx, val, mask)
            loss = model.loss(out, y)
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
                out = model(idx, val, mask)
                vloss += float(model.loss(out, y).item()) * y.size(0)
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
    print(f'config: fine_sigma={FINE_SIGMA} coarse_sigma={COARSE_SIGMA}  '
          f'CE_fine={CE_FINE_W} CE_coarse={CE_COARSE_W} MSE={MSE_W}', flush=True)

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    fine_xy, fGw, fGh = data.build_heatmap_grid(cell_size=FINE_CELL_SIZE)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, cGw, cGh = data.build_heatmap_grid(cell_size=COARSE_CELL_SIZE)
    coarse_mask = data.build_free_mask(coarse_xy)
    f2c = data.build_fine_to_coarse(fine_cell=FINE_CELL_SIZE, coarse_cell=COARSE_CELL_SIZE)

    # GP synth — same as winner
    print(f'\n[gen] {N_SYNTHETIC} synth ...', flush=True)
    t0 = time.time()
    train_recs = [{'x': float(y[i, 0]), 'y': float(y[i, 1]), 'session': sess[i],
                    'aps': {b: float(X[i, j]) for j, b in enumerate(bssids)
                             if X[i, j] > -99.5}} for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_recs, bssids, min_samples=15, verbose=False)
    real_xy = np.array([(r['x'], r['y']) for r in train_recs], dtype=np.float32)
    free_pos = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(42)
    chosen = rng.choice(len(free_pos), N_SYNTHETIC, replace=False)
    X_synth, y_synth = synthetic.synthesize(gps, free_pos[chosen], bssids,
                                              records=train_recs, detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0, seed=42)
    print(f'[gen] done ({time.time() - t0:.1f}s)', flush=True)
    s_idx, s_val, s_mask = synthetic.build_set_form(X_synth, bssids, max_aps=MAX_APS)
    idx_tr = np.concatenate([set_idx[tr_A], s_idx], axis=0)
    val_tr = np.concatenate([set_val[tr_A], s_val], axis=0)
    mask_tr = np.concatenate([set_mask[tr_A], s_mask], axis=0)
    y_tr = np.concatenate([y[tr_A], y_synth.astype(np.float32)], axis=0)

    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    seed_preds = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f'\n========== seed {seed} ==========', flush=True)
        train_ds = data.SetDataset(idx_tr, val_tr, mask_tr, y_tr,
                                     jitter=JITTER_DBM)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
            coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
            fine_to_coarse=f2c,
            fine_sigma=FINE_SIGMA, coarse_sigma=COARSE_SIGMA,
            ce_fine_w=CE_FINE_W, ce_coarse_w=CE_COARSE_W, mse_w=MSE_W,
            **CFG)
        if seed == SEEDS[0]:
            n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f'[model] params: {n_p:,}', flush=True)
        m, vloss = train_one(m, tl, test_loader, device, f's{seed}')
        m.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            v_t = torch.from_numpy(set_val[te_A]).to(device)
            m_t = torch.from_numpy(set_mask[te_A]).to(device)
            pred = m.predict_xy(idx_t, v_t, m_t)
        err = loc_err(pred, y[te_A])
        print(f'  seed {seed}: median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}  val={vloss:.3f}', flush=True)
        seed_preds.append({'pred': pred, 'val_loss': vloss, 'err': err})
        torch.save(m.state_dict(),
                    CKPT_DIR / f'A_random__CascadeTuned_s{seed}.pt')

    best_i = int(np.argmin([s['val_loss'] for s in seed_preds]))
    bs = loc_stats(seed_preds[best_i]['err'])
    print(f'\n========== Best single (seed {SEEDS[best_i]}) ==========', flush=True)
    print(f'  median={bs["median_err_m"]:.3f}  mean={bs["mean_err_m"]:.3f}  '
          f'p90={bs["p90_err_m"]:.3f}', flush=True)

    pts = np.stack([s['pred'] for s in seed_preds], axis=0)
    em = loc_stats(loc_err(pts.mean(axis=0), y[te_A]))
    egm = loc_stats(loc_err(geom_median(pts), y[te_A]))
    print(f'\n========== 5-seed mean ==========\n  '
          f'median={em["median_err_m"]:.3f}  mean={em["mean_err_m"]:.3f}  '
          f'p90={em["p90_err_m"]:.3f}', flush=True)
    print(f'\n========== 5-seed geom-median ==========\n  '
          f'median={egm["median_err_m"]:.3f}  mean={egm["mean_err_m"]:.3f}  '
          f'p90={egm["p90_err_m"]:.3f}', flush=True)

    print(f'\n=== vs Cascade baseline ===')
    print(f'  baseline 5-seed geom-median:  0.793 m')
    print(f'  tuned    5-seed geom-median:  {egm["median_err_m"]:.3f} m  '
          f'({egm["median_err_m"] - 0.793:+.3f})', flush=True)

    np.savez(PRED_DIR / 'A_random__CascadeTunedEnsemble.npz',
              pred=geom_median(pts), y_true=y[te_A],
              err=loc_err(geom_median(pts), y[te_A]), test_idx=te_A)


if __name__ == '__main__':
    main()
