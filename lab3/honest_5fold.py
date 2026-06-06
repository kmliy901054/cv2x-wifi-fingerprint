"""5-fold cross-validation honest pipeline.

Build a much more reliable validation signal than the single 144-sample split
by using 5-fold cross-validation:

  For each fold k (k=0..4):
    val_k = samples in fold k
    db_k  = samples in folds {0..4} \ {k}
    For each EmbKNN variant, predict on val_k using db_k as the KNN database.
  Concatenate val predictions across all 5 folds → 1449 'cross-validated' val
  predictions per EmbKNN config. This is a much more reliable selection signal
  than a single 144-sample split.

  For each Cascade decoder variant, predictions on val_k are computed via the
  encoder trained on the FULL train set (encoder leakage — we don't retrain
  per fold because that would cost 5× the GPU budget; we acknowledge this).

  Greedy + swap search on the 1449 val predictions → pick combo.
  Final: re-evaluate that combo on test using the FULL train as the KNN db.

If even this gives > 0.65 on test, our true generalization is honestly above
0.65 and we cannot claim it without genuine model improvements.
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
PRED_DIR = HERE / 'outputs' / 'predictions'
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
N_FOLDS = 5

CASCADE_VARIANTS = ['Cascade', 'CascadeTuned', 'CascadeAggressive', 'CascadeBig']
# (encoder_name, K, weight_spec, save_name)
EMBKNN_HEADS = [
    ('Cascade',          5, 'inv',           'EmbKNN_BaseK5'),
    ('Cascade',          7, 'inv',           'EmbKNN_BaseK7'),
    ('Cascade',         10, 'inv',           'EmbKNN_BaseK10'),
    ('Cascade',          3, 'inv',           'EmbKNN_BaseK3'),
    ('Cascade',          5, ('soft', 0.1),   'EmbKNN_BaseSoftK5T01'),
    ('Cascade',          7, ('soft', 0.2),   'EmbKNN_BaseSoftK7T02'),
    ('Cascade',          5, ('soft', 0.15),  'EmbKNN_BSoftK5T15'),
    ('Cascade',          7, ('soft', 0.1),   'EmbKNN_BSoftK7T10'),
    ('Cascade',          8, ('soft', 0.08),  'EmbKNN_BSoftK8T08'),
    ('Cascade',         10, ('soft', 0.1),   'EmbKNN_BSoftK10T10'),
    ('CascadeTuned',     5, 'inv',           'EmbKNN_Tuned'),
    ('CascadeTuned',     7, 'inv',           'EmbKNN_TunedK7'),
    ('CascadeTuned',    10, 'inv',           'EmbKNN_TunedK10'),
    ('CascadeTuned',     5, ('soft', 0.1),   'EmbKNN_TunedSoftK5T01'),
    ('CascadeTuned',     5, ('soft', 0.05),  'EmbKNN_TSoftK5T05'),
    ('CascadeTuned',    10, ('soft', 0.1),   'EmbKNN_TSoftK10T10'),
    ('CascadeAggressive', 5, 'inv',          'EmbKNN_AggrK5'),
    ('CascadeAggressive', 5, ('soft', 0.1),  'EmbKNN_AggrSoftK5T01'),
]


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d; w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps: break
        m = new
    return m


def med(p, yt): return float(np.median(np.linalg.norm(p - yt, axis=1)))
def stats(p, yt):
    e = np.linalg.norm(p - yt, axis=1)
    return (float(np.median(e)), float(e.mean()), float(np.percentile(e, 90)))


def cascade_preds_for(name, idx_q, val_q, mask_q, **k):
    preds = []
    for s in range(42, 60):
        ck = CKPT_DIR / f'A_random__{name}_s{s}.pt'
        if not ck.exists(): continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(k['bssids']),
            fine_cell_xy=k['fxy'], fine_free_mask=k['fmask'].astype(np.float32),
            coarse_cell_xy=k['cxy'], coarse_free_mask=k['cmask'].astype(np.float32),
            fine_to_coarse=k['f2c'], **CFG).to('cpu')
        m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval()
        with torch.no_grad():
            it = torch.from_numpy(idx_q); vt = torch.from_numpy(val_q); mt = torch.from_numpy(mask_q)
            preds.append(m.predict_xy(it, vt, mt))
    return np.stack(preds, axis=0) if preds else None


def encoder_embeddings(name, idx_q, val_q, mask_q, **k):
    Z_acc = None; n = 0
    for s in range(42, 60):
        ck = CKPT_DIR / f'A_random__{name}_s{s}.pt'
        if not ck.exists(): continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(k['bssids']),
            fine_cell_xy=k['fxy'], fine_free_mask=k['fmask'].astype(np.float32),
            coarse_cell_xy=k['cxy'], coarse_free_mask=k['cmask'].astype(np.float32),
            fine_to_coarse=k['f2c'], **CFG).to('cpu')
        m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval()
        with torch.no_grad():
            out = []
            for st in range(0, idx_q.shape[0], 256):
                en = min(st + 256, idx_q.shape[0])
                o = m.encode(torch.from_numpy(idx_q[st:en]),
                              torch.from_numpy(val_q[st:en]),
                              torch.from_numpy(mask_q[st:en])).numpy()
                out.append(o)
            Z = np.concatenate(out, axis=0)
        Z_acc = Z if Z_acc is None else Z_acc + Z
        n += 1
    if n == 0: return None
    Z = Z_acc / n
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def knn(Z_db, y_db, Z_q, K, weight_spec):
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_db)
    d, idx = nn.kneighbors(Z_q)
    if weight_spec == 'inv':
        w = 1.0 / (d + 1e-6)
    else:
        _, tau = weight_spec
        w = np.exp(-d / tau)
    w = w / w.sum(axis=1, keepdims=True)
    return (w[..., None] * y_db[idx]).sum(axis=1)


def greedy_with_swap(cand_preds, yt):
    """Multi-start greedy + swap. cand_preds: dict[name] = (n_seeds, T, 2)."""
    best_global = None
    for start in cand_preds:
        sel = [start]
        # forward greedy
        while True:
            cur = med(geom_median(np.concatenate([cand_preds[s] for s in sel], axis=0)), yt)
            best_add = None; best_m = cur
            for n in cand_preds:
                if n in sel: continue
                stk = np.concatenate([cand_preds[s] for s in sel] + [cand_preds[n]], axis=0)
                m = med(geom_median(stk), yt)
                if m < best_m - 1e-4:
                    best_m = m; best_add = n
            if best_add is None: break
            sel.append(best_add)
        # swap
        improved = True
        while improved:
            improved = False
            cur = med(geom_median(np.concatenate([cand_preds[s] for s in sel], axis=0)), yt)
            for i in range(len(sel)):
                for n in cand_preds:
                    if n in sel: continue
                    trial = sel.copy(); trial[i] = n
                    m = med(geom_median(np.concatenate([cand_preds[s] for s in trial], axis=0)), yt)
                    if m < cur - 1e-4:
                        sel = trial; cur = m; improved = True; break
                if improved: break
        final_med = med(geom_median(np.concatenate([cand_preds[s] for s in sel], axis=0)), yt)
        if best_global is None or final_med < best_global[1]:
            best_global = (sel, final_med)
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

    # Stratified-ish 5-fold split of tr_A
    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(tr_A))
    folds = np.array_split(perm, N_FOLDS)
    print(f'train_A: {len(tr_A)} samples  →  5 folds of size {len(folds[0])}±')

    # Pre-compute encoder embeddings for full train and test (constant)
    print('\n--- encoder embeddings on full train + test ---')
    embeds_tr = {}; embeds_te = {}
    for enc in CASCADE_VARIANTS:
        embeds_tr[enc] = encoder_embeddings(enc,
                                              set_idx[tr_A], set_val[tr_A],
                                              set_mask[tr_A], **kw)
        embeds_te[enc] = encoder_embeddings(enc,
                                              set_idx[te_A], set_val[te_A],
                                              set_mask[te_A], **kw)
        print(f'  {enc}: train {embeds_tr[enc].shape if embeds_tr[enc] is not None else "None"}')

    # === Compute val predictions across 5 folds ===
    print('\n--- 5-fold val predictions ---')
    # For Cascade decoders: predict on all train (encoder LEAKS — known caveat)
    val_preds = {}
    for v in CASCADE_VARIANTS:
        p = cascade_preds_for(v, set_idx[tr_A], set_val[tr_A], set_mask[tr_A], **kw)
        if p is not None:
            val_preds[v] = p
            print(f'  {v}: val preds {p.shape}  (encoder-leaky)')

    # For EmbKNN heads: HONEST — train_inner = other folds, val = this fold
    for enc, K, wt, name in EMBKNN_HEADS:
        if embeds_tr[enc] is None: continue
        Z_full = embeds_tr[enc]
        preds_per_fold = np.zeros((len(tr_A), 2), dtype=np.float32)
        for fk in range(N_FOLDS):
            val_local = folds[fk]
            db_local = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fk])
            Z_db = Z_full[db_local]
            Z_q = Z_full[val_local]
            y_db = y[tr_A[db_local]]
            preds_per_fold[val_local] = knn(Z_db, y_db, Z_q, K, wt)
        val_preds[name] = preds_per_fold[None, ...]                # (1, 1449, 2)
        e = np.linalg.norm(preds_per_fold - y[tr_A], axis=1)
        print(f'  {name}: 5-fold cv median {np.median(e):.3f}')

    yt_val = y[tr_A]                                                # 1449 honest val labels

    # === Greedy + swap on 1449 honest val predictions ===
    print('\n--- greedy + swap on 1449 5-fold val ---')
    sel, val_med = greedy_with_swap(val_preds, yt_val)
    print(f'  selected: {sel}')
    print(f'  val median: {val_med:.3f}  (over 1449 samples)')

    # === Final test eval with selected combo ===
    print('\n--- final test evaluation with the SELECTED combo ---')
    test_preds = {}
    for v in CASCADE_VARIANTS:
        if v in sel:
            test_preds[v] = cascade_preds_for(v, set_idx[te_A], set_val[te_A],
                                                 set_mask[te_A], **kw)
    for enc, K, wt, name in EMBKNN_HEADS:
        if name in sel:
            Z_db = embeds_tr[enc]; Z_q = embeds_te[enc]
            p = knn(Z_db, y[tr_A], Z_q, K, wt)
            test_preds[name] = p[None, ...]

    stacked = np.concatenate([test_preds[s] for s in sel], axis=0)
    pred = geom_median(stacked)
    m_t, mn_t, p9_t = stats(pred, y[te_A])
    print(f'  test median = {m_t:.3f}  (HONEST estimate, no test leakage)')
    print(f'  test mean   = {mn_t:.3f}')
    print(f'  test p90    = {p9_t:.3f}')
    print(f'  within 0.3 m: {(np.linalg.norm(pred - y[te_A], axis=1) <= 0.3).mean() * 100:.1f}%')

    # Save honest result
    err = np.linalg.norm(pred - y[te_A], axis=1)
    np.savez(PRED_DIR / 'A_random__Honest5Fold.npz',
              pred=pred, y_true=y[te_A], err=err, test_idx=te_A,
              members=np.array(sel))
    print(f'\n[save] -> predictions/A_random__Honest5Fold.npz')


if __name__ == '__main__':
    main()
