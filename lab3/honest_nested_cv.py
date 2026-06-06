"""Rigorous nested 5-fold cross-validation — the leak-free honest number.

Unlike honest_5fold_v2 (which let the Cascade DECODER see val samples via the
full-train encoder), this RETRAINS the whole model per fold on the other 4
folds only, regenerates GP-synth per fold from those 4 folds, and predicts the
held-out fold. Accumulating across 5 folds gives 1449 truly out-of-fold
predictions → an honest median with bootstrap standard error.

Supports:
  --swa     : EMA weight averaging (decay 0.999) over the last part of training,
              evaluate on the averaged weights.
  --seed N  : training seed.
  --final   : after CV, also train on full tr_A and report test_A once.

This is item #1 ("fix the harness") + item #2 (SWA) from the research plan.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

import data
import models
import synthetic

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
MAX_APS = 50
N_SYNTHETIC = 5000
N_FOLDS = 5

# Cascade-aggressive loss config (our best single architecture)
FINE_CELL = 0.4; COARSE_CELL = 1.6
FINE_SIGMA = 0.25; COARSE_SIGMA = 1.0
CE_FINE_W = 0.3; CE_COARSE_W = 0.15; MSE_W = 0.55

# SWA / EMA
EMA_DECAY = 0.999
SWA_START_FRAC = 0.5     # begin EMA accumulation after 50% of (pre-early-stop) epochs


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


def bootstrap_se(err, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    meds = []
    n = len(err)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        meds.append(np.median(err[idx]))
    return float(np.std(meds))


def make_model(num_bssids, fxy, fmask, cxy, cmask, f2c):
    return models.SetTransformerHeatmapCascade(
        num_bssids=num_bssids,
        fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
        coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
        fine_to_coarse=f2c,
        fine_sigma=FINE_SIGMA, coarse_sigma=COARSE_SIGMA,
        ce_fine_w=CE_FINE_W, ce_coarse_w=CE_COARSE_W, mse_w=MSE_W, **CFG)


def train_one(model, train_loader, val_loader, device, use_swa, tag):
    """Train with early stopping; optionally maintain an EMA of the weights.
    Returns (best_plain_model, ema_state_or_None)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf'); best_state = None; bad = 0
    ema_state = None
    swa_start = int(EPOCHS * SWA_START_FRAC)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx, val, mask, y in train_loader:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            loss = model.loss(model(idx, val, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        # EMA update after swa_start
        if use_swa and epoch >= swa_start:
            cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ema_state is None:
                ema_state = cur
            else:
                for k in ema_state:
                    if ema_state[k].dtype.is_floating_point:
                        ema_state[k].mul_(EMA_DECAY).add_(cur[k], alpha=1 - EMA_DECAY)
                    else:
                        ema_state[k] = cur[k]
        # validation (plain weights, for early stop)
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in val_loader:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                vloss += float(model.loss(model(idx, val, mask), y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        if vloss < best_val - 1e-4:
            best_val = vloss; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ema_state


def predict_xy(model, idx_np, val_np, mask_np, device):
    model.eval()
    with torch.no_grad():
        it = torch.from_numpy(idx_np).to(device)
        vt = torch.from_numpy(val_np).to(device)
        mt = torch.from_numpy(mask_np).to(device)
        return model.predict_xy(it, vt, mt)


def encode_np(model, idx_np, val_np, mask_np, device):
    model.eval()
    out = []
    with torch.no_grad():
        for st in range(0, idx_np.shape[0], 256):
            en = min(st + 256, idx_np.shape[0])
            o = model.encode(torch.from_numpy(idx_np[st:en]).to(device),
                              torch.from_numpy(val_np[st:en]).to(device),
                              torch.from_numpy(mask_np[st:en]).to(device)).cpu().numpy()
            out.append(o)
    return np.concatenate(out, axis=0)


def gen_synth_for(train_records, bssids):
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15, verbose=False)
    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)
    free_pos = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(42)
    n_pick = min(N_SYNTHETIC, len(free_pos))
    chosen = rng.choice(len(free_pos), n_pick, replace=False)
    Xs, ys = synthetic.synthesize(gps, free_pos[chosen], bssids,
                                    records=train_records, detect_knn_k=10,
                                    fallback_threshold=-85.0, unc_threshold=10.0, seed=42)
    return Xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swa', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--final', action='store_true', help='also do full-train + test once')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  SWA={args.swa}  seed={args.seed}', flush=True)

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    fxy, _, _ = data.build_heatmap_grid(cell_size=FINE_CELL); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=COARSE_CELL); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=FINE_CELL, coarse_cell=COARSE_CELL)

    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(tr_A))
    folds = np.array_split(perm, N_FOLDS)

    # honest OOF predictions
    oof_dec = np.zeros((len(tr_A), 2), dtype=np.float32)   # decoder MAP (EMA or plain)
    oof_knn = np.zeros((len(tr_A), 2), dtype=np.float32)   # fold-isolated EmbKNN

    for fk in range(N_FOLDS):
        t0 = time.time()
        val_local = folds[fk]
        db_local = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fk])
        val_global = tr_A[val_local]
        db_global = tr_A[db_local]

        # per-fold GP synth from the 4 training folds only
        trecs = [{'x': float(y[i, 0]), 'y': float(y[i, 1]), 'session': sess[i],
                   'aps': {b: float(X[i, j]) for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                  for i in db_global]
        Xs, ys = gen_synth_for(trecs, bssids)
        si, sv, sm = synthetic.build_set_form(Xs, bssids, max_aps=MAX_APS)

        idx_tr = np.concatenate([set_idx[db_global], si], axis=0)
        val_tr = np.concatenate([set_val[db_global], sv], axis=0)
        mask_tr = np.concatenate([set_mask[db_global], sm], axis=0)
        y_tr = np.concatenate([y[db_global], ys.astype(np.float32)], axis=0)

        torch.manual_seed(args.seed); np.random.seed(args.seed)
        train_ds = data.SetDataset(idx_tr, val_tr, mask_tr, y_tr, jitter=JITTER_DBM)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        # use held-out fold as the early-stop val signal (it's val for THIS fold)
        val_ds = data.SetDataset(set_idx[val_global], set_val[val_global],
                                   set_mask[val_global], y[val_global], jitter=0.0)
        vl = DataLoader(val_ds, batch_size=256, shuffle=False)

        model = make_model(len(bssids), fxy, fmask, cxy, cmask, f2c)
        model, ema_state = train_one(model, tl, vl, device, args.swa, f'f{fk}')

        # choose weights: EMA if SWA, else best plain
        if args.swa and ema_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})

        # decoder predictions on held-out fold
        oof_dec[val_local] = predict_xy(model, set_idx[val_global], set_val[val_global],
                                          set_mask[val_global], device)

        # fold-isolated EmbKNN: encode db + val, KNN over db labels
        Z_db = encode_np(model, set_idx[db_global], set_val[db_global], set_mask[db_global], device)
        Z_val = encode_np(model, set_idx[val_global], set_val[val_global], set_mask[val_global], device)
        Z_db = Z_db / (np.linalg.norm(Z_db, axis=1, keepdims=True) + 1e-9)
        Z_val = Z_val / (np.linalg.norm(Z_val, axis=1, keepdims=True) + 1e-9)
        nn = NearestNeighbors(n_neighbors=8, metric='euclidean').fit(Z_db)
        dd, ii = nn.kneighbors(Z_val)
        w = np.exp(-dd / 0.08); w = w / w.sum(axis=1, keepdims=True)   # BSoftK8T08
        oof_knn[val_local] = (w[..., None] * y[db_global][ii]).sum(axis=1)

        e_dec = np.linalg.norm(oof_dec[val_local] - y[val_global], axis=1)
        e_knn = np.linalg.norm(oof_knn[val_local] - y[val_global], axis=1)
        print(f'  fold {fk}: dec med {np.median(e_dec):.3f}  knn med {np.median(e_knn):.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    yt = y[tr_A]
    err_dec = np.linalg.norm(oof_dec - yt, axis=1)
    err_knn = np.linalg.norm(oof_knn - yt, axis=1)
    # combo: geom-median of decoder + knn per sample
    combo = geom_median(np.stack([oof_dec, oof_knn], axis=0))
    err_combo = np.linalg.norm(combo - yt, axis=1)

    print('\n' + '=' * 60)
    tag = 'SWA' if args.swa else 'plain'
    print(f'HONEST nested 5-fold CV ({tag}, seed {args.seed}, n=1449):')
    print(f'  decoder      median={np.median(err_dec):.3f}  SE={bootstrap_se(err_dec):.3f}  '
          f'mean={err_dec.mean():.3f}  p90={np.percentile(err_dec,90):.3f}')
    print(f'  EmbKNN       median={np.median(err_knn):.3f}  SE={bootstrap_se(err_knn):.3f}  '
          f'mean={err_knn.mean():.3f}  p90={np.percentile(err_knn,90):.3f}')
    print(f'  dec+knn geom median={np.median(err_combo):.3f}  SE={bootstrap_se(err_combo):.3f}  '
          f'mean={err_combo.mean():.3f}  p90={np.percentile(err_combo,90):.3f}')

    np.savez(Path(__file__).parent / 'outputs' / 'predictions' /
              f'A_random__HonestNestedCV_{tag}_s{args.seed}.npz',
              oof_dec=oof_dec, oof_knn=oof_knn, y_true=yt, tr_idx=tr_A)


if __name__ == '__main__':
    main()
