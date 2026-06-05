"""Train 5 more seeds (47-51) of Cascade-tuned to grow ensemble from 5 → 10.

Same exact hyperparams as train_cascade_tuned.py — just different seeds.
For the mega-ensemble: more samples = more stable geom-median.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import data, models, synthetic

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PRED_DIR = HERE / 'outputs' / 'predictions'
CKPT_DIR = HERE / 'outputs' / 'checkpoints'

CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
WEIGHT_DECAY = 1e-3; JITTER_DBM = 4.0; BATCH_SIZE = 64
LR = 1e-3; EPOCHS = 300; PATIENCE = 50
SEEDS = [47, 48, 49, 50, 51]                            # ← extend ensemble
MAX_APS = 50; MIN_BSSID_COUNT = 10; N_SYNTHETIC = 5000

# IDENTICAL hyperparams to train_cascade_tuned.py
FINE_SIGMA = 0.3; COARSE_SIGMA = 1.0
CE_FINE_W = 0.4; CE_COARSE_W = 0.2; MSE_W = 0.4


def loc_err(yp, yt): return np.linalg.norm(yp - yt, axis=1)


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d; w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps: break
        m = new
    return m


def train_one(model, tl, vl, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = float('inf'); best_s = None; bad = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for idx, val, mask, y in tl:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            out = model(idx, val, mask); loss = model.loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        vl_sum = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in vl:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                out = model(idx, val, mask)
                vl_sum += float(model.loss(out, y).item()) * y.size(0); vn += y.size(0)
        vloss = vl_sum / max(1, vn)
        if vloss < best - 1e-4:
            best = vloss; bad = 0
            best_s = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  [{tag}] early stop @ epoch {ep}, best val = {best:.4f}', flush=True)
                break
        if ep == 1 or ep % 40 == 0:
            print(f'  [{tag}] epoch {ep:3d}  val_loss {vloss:.4f}  best {best:.4f}', flush=True)
    if best_s is not None: model.load_state_dict(best_s)
    return model, best


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}    seeds: {SEEDS}', flush=True)

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    print(f'\n[gen] {N_SYNTHETIC} synth ...', flush=True)
    t0 = time.time()
    trecs = [{'x': float(y[i, 0]), 'y': float(y[i, 1]), 'session': sess[i],
              'aps': {b: float(X[i, j]) for j, b in enumerate(bssids) if X[i, j] > -99.5}}
             for i in tr_A]
    gps = synthetic.fit_per_ap_gps(trecs, bssids, min_samples=15, verbose=False)
    real_xy = np.array([(r['x'], r['y']) for r in trecs], dtype=np.float32)
    free_pos = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(42)
    chosen = rng.choice(len(free_pos), N_SYNTHETIC, replace=False)
    Xs, ys = synthetic.synthesize(gps, free_pos[chosen], bssids, records=trecs,
                                    detect_knn_k=10, fallback_threshold=-85.0,
                                    unc_threshold=10.0, seed=42)
    print(f'[gen] done ({time.time() - t0:.1f}s)', flush=True)
    si, sv, sm = synthetic.build_set_form(Xs, bssids, max_aps=MAX_APS)
    idx_tr = np.concatenate([set_idx[tr_A], si], axis=0)
    val_tr = np.concatenate([set_val[tr_A], sv], axis=0)
    mask_tr = np.concatenate([set_mask[tr_A], sm], axis=0)
    y_tr = np.concatenate([y[tr_A], ys.astype(np.float32)], axis=0)

    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    tl_v = DataLoader(test_ds, batch_size=256, shuffle=False)

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f'\n========== seed {seed} ==========', flush=True)
        train_ds = data.SetDataset(idx_tr, val_tr, mask_tr, y_tr, jitter=JITTER_DBM)
        tl_t = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c,
            fine_sigma=FINE_SIGMA, coarse_sigma=COARSE_SIGMA,
            ce_fine_w=CE_FINE_W, ce_coarse_w=CE_COARSE_W, mse_w=MSE_W, **CFG)
        m, vloss = train_one(m, tl_t, tl_v, device, f's{seed}')
        m.eval()
        with torch.no_grad():
            it = torch.from_numpy(set_idx[te_A]).to(device)
            vt = torch.from_numpy(set_val[te_A]).to(device)
            mt = torch.from_numpy(set_mask[te_A]).to(device)
            pred = m.predict_xy(it, vt, mt)
        err = loc_err(pred, y[te_A])
        print(f'  seed {seed}: median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}  val={vloss:.3f}', flush=True)
        torch.save(m.state_dict(),
                    CKPT_DIR / f'A_random__CascadeTuned_s{seed}.pt')


if __name__ == '__main__':
    main()
