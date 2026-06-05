"""Alternative models to add diversity to the Cascade mega-ensemble.

Two non-neural / hybrid approaches with very different inductive biases:

  GB:     sklearn GradientBoostingRegressor on dense (X, presence-mask) →
          two independent regressors for x and y. Different feature interaction
          patterns than the Set Transformer.

  EmbKNN: use the trained Cascade-tuned encoder as a fixed feature extractor;
          embed each train scan to 192-d; at inference, KNN over training
          embeddings and inverse-distance-weighted average their (x, y).
          Retrieval-based — gives accurate predictions in densely-sampled
          regions of train space.

Both are fast to fit (minutes on CPU) and add a fundamentally different
prediction profile vs neural decoders.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

import data, models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / 'outputs' / 'checkpoints'
PRED_DIR = HERE / 'outputs' / 'predictions'
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
SEEDS = [42, 43, 44, 45, 46]


def loc_err(yp, yt): return np.linalg.norm(yp - yt, axis=1)


def main():
    # GPU may be busy; both fits are fast on CPU
    device = torch.device('cpu')
    print(f'device: {device}')

    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']

    # ── 1. GradientBoosting on (RSSI, presence mask) ──────────────
    print('\n=== GradientBoosting (dense RSSI + mask) ===')
    X_tr = X[tr_A]
    X_te = X[te_A]
    mask_tr = (X_tr > -99.5).astype(np.float32)
    mask_te = (X_te > -99.5).astype(np.float32)
    feat_tr = np.concatenate([X_tr, mask_tr], axis=1)
    feat_te = np.concatenate([X_te, mask_te], axis=1)

    t0 = time.time()
    gb_x = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
    gb_y = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
    gb_x.fit(feat_tr, y[tr_A, 0])
    gb_y.fit(feat_tr, y[tr_A, 1])
    pred_gb = np.stack([gb_x.predict(feat_te), gb_y.predict(feat_te)], axis=1)
    err = loc_err(pred_gb, y[te_A])
    print(f'  fit time: {time.time() - t0:.1f}s')
    print(f'  median = {np.median(err):.3f}   mean = {err.mean():.3f}   '
          f'p90 = {np.percentile(err, 90):.3f}')
    np.savez(PRED_DIR / 'A_random__GB.npz',
              pred=pred_gb, y_true=y[te_A], err=err, test_idx=te_A)

    # ── 2. EmbKNN: Cascade-tuned encoder → KNN on embedding ───────
    print('\n=== EmbKNN (Cascade-tuned encoder → KNN) ===')
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)
    fine_xy, _, _ = data.build_heatmap_grid(cell_size=0.4)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
    coarse_mask = data.build_free_mask(coarse_xy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    # average embeddings across the 5 tuned seeds (more robust feature)
    Z_tr_acc = None; Z_te_acc = None; n_used = 0
    for s in SEEDS:
        ck = CKPT_DIR / f'A_random__CascadeTuned_s{s}.pt'
        if not ck.exists():
            print(f'  [skip] missing {ck.name}'); continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
            coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to(device)
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            def emb(idx_a, val_a, mask_a):
                B = idx_a.shape[0]
                out = []
                for st in range(0, B, 256):
                    en = min(st + 256, B)
                    it = torch.from_numpy(idx_a[st:en]).to(device)
                    vt = torch.from_numpy(val_a[st:en]).to(device)
                    mt = torch.from_numpy(mask_a[st:en]).to(device)
                    out.append(m.encode(it, vt, mt).cpu().numpy())
                return np.concatenate(out, axis=0)
            Z_tr = emb(set_idx[tr_A], set_val[tr_A], set_mask[tr_A])
            Z_te = emb(set_idx[te_A], set_val[te_A], set_mask[te_A])
        Z_tr_acc = Z_tr if Z_tr_acc is None else Z_tr_acc + Z_tr
        Z_te_acc = Z_te if Z_te_acc is None else Z_te_acc + Z_te
        n_used += 1
    Z_tr = Z_tr_acc / n_used; Z_te = Z_te_acc / n_used
    # L2-normalize for cosine-ish KNN
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-9)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-9)
    for K in [3, 5, 7, 10]:
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_tr)
        d, idx = nn.kneighbors(Z_te)
        w = 1.0 / (d + 1e-6); w = w / w.sum(axis=1, keepdims=True)
        pred = (w[..., None] * y[tr_A][idx]).sum(axis=1)
        err = loc_err(pred, y[te_A])
        print(f'  K={K}: median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}')
        if K == 5:
            np.savez(PRED_DIR / 'A_random__EmbKNN.npz',
                      pred=pred, y_true=y[te_A], err=err, test_idx=te_A)


if __name__ == '__main__':
    main()
