"""Honest pipeline v2: EmbKNN-only val selection (avoid Cascade encoder leak).

The Cascade decoder predictions on training data are encoder-leaky (the
encoder was trained on those samples) — so v1's greedy picked them
artificially. EmbKNN predictions ARE honest because we used 5-fold KNN
database isolation.

This v2:
  1. Compute 5-fold honest val predictions for EmbKNN heads only.
  2. Greedy+swap on EmbKNN-only val → pick best EmbKNN combo.
  3. At test time, evaluate:
       (a) the chosen EmbKNN combo alone
       (b) plus each Cascade variant added (one at a time, picked by val
           encoder-leaky signal — but at test we honestly report the result)
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import data, models
from honest_5fold import (
    CASCADE_VARIANTS, EMBKNN_HEADS, geom_median, med, stats,
    cascade_preds_for, encoder_embeddings, knn,
    N_FOLDS, CFG, CKPT_DIR,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PRED_DIR = HERE / 'outputs' / 'predictions'


def greedy_swap(cand, yt):
    best_global = None
    for start in cand:
        sel = [start]
        while True:
            cur = med(geom_median(np.concatenate([cand[s] for s in sel], axis=0)), yt)
            add = None; bm = cur
            for n in cand:
                if n in sel: continue
                stk = np.concatenate([cand[s] for s in sel] + [cand[n]], axis=0)
                m = med(geom_median(stk), yt)
                if m < bm - 1e-4:
                    bm = m; add = n
            if add is None: break
            sel.append(add)
        improved = True
        while improved:
            improved = False
            cur = med(geom_median(np.concatenate([cand[s] for s in sel], axis=0)), yt)
            for i in range(len(sel)):
                for n in cand:
                    if n in sel: continue
                    trial = sel.copy(); trial[i] = n
                    m = med(geom_median(np.concatenate([cand[s] for s in trial], axis=0)), yt)
                    if m < cur - 1e-4:
                        sel = trial; cur = m; improved = True; break
                if improved: break
        fm = med(geom_median(np.concatenate([cand[s] for s in sel], axis=0)), yt)
        if best_global is None or fm < best_global[1]:
            best_global = (sel, fm)
    return best_global


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)
    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)
    kw = dict(bssids=bssids, fxy=fxy, fmask=fmask, cxy=cxy, cmask=cmask, f2c=f2c)
    yt_val = y[tr_A]; yt_te = y[te_A]

    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(tr_A))
    folds = np.array_split(perm, N_FOLDS)

    # encoder embeddings on full train + test
    embeds_tr = {}; embeds_te = {}
    for enc in CASCADE_VARIANTS:
        embeds_tr[enc] = encoder_embeddings(enc, set_idx[tr_A], set_val[tr_A],
                                              set_mask[tr_A], **kw)
        embeds_te[enc] = encoder_embeddings(enc, set_idx[te_A], set_val[te_A],
                                              set_mask[te_A], **kw)

    # EmbKNN-only 5-fold val predictions
    val_preds_emb = {}
    for enc, K, wt, name in EMBKNN_HEADS:
        if embeds_tr[enc] is None: continue
        Z_full = embeds_tr[enc]
        preds = np.zeros((len(tr_A), 2), dtype=np.float32)
        for fk in range(N_FOLDS):
            val_local = folds[fk]
            db_local = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fk])
            preds[val_local] = knn(Z_full[db_local], y[tr_A[db_local]],
                                     Z_full[val_local], K, wt)
        val_preds_emb[name] = preds[None, ...]
    print(f'EmbKNN heads available: {len(val_preds_emb)}')

    # Greedy on EmbKNN-only val
    sel, val_med = greedy_swap(val_preds_emb, yt_val)
    print(f'\nEmbKNN-only val selected: {sel}')
    print(f'  honest val median: {val_med:.3f}  (1449 samples)')

    # Test eval: build EmbKNN preds on test
    test_preds_emb = {}
    for enc, K, wt, name in EMBKNN_HEADS:
        if name not in sel: continue
        p = knn(embeds_tr[enc], y[tr_A], embeds_te[enc], K, wt)
        test_preds_emb[name] = p[None, ...]

    stacked = np.concatenate([test_preds_emb[s] for s in sel], axis=0)
    p = geom_median(stacked)
    m_t, mn_t, p9_t = stats(p, yt_te)
    print(f'\n=== EmbKNN-only honest result ===')
    print(f'  test median = {m_t:.3f}   mean = {mn_t:.3f}   p90 = {p9_t:.3f}')
    print(f'  within 0.3 m: {(np.linalg.norm(p - yt_te, axis=1) <= 0.3).mean() * 100:.1f}%')

    # === Add each Cascade variant and report honestly ===
    print('\n--- Adding Cascade decoders to the selected EmbKNN combo ---')
    test_cascade = {}
    for v in CASCADE_VARIANTS:
        test_cascade[v] = cascade_preds_for(v, set_idx[te_A], set_val[te_A],
                                                set_mask[te_A], **kw)

    for v in CASCADE_VARIANTS:
        stk = np.concatenate([test_cascade[v]] +
                              [test_preds_emb[s] for s in sel], axis=0)
        p = geom_median(stk); m, mn, p9 = stats(p, yt_te)
        print(f'  +{v:<22}  test median={m:.3f}  mean={mn:.3f}  p90={p9:.3f}')

    # Best EmbKNN + all Cascade
    stk = np.concatenate(list(test_cascade.values()) +
                          [test_preds_emb[s] for s in sel], axis=0)
    p = geom_median(stk); m, mn, p9 = stats(p, yt_te)
    print(f'  +ALL Cascade variants  test median={m:.3f}  mean={mn:.3f}  p90={p9:.3f}')


if __name__ == '__main__':
    main()
