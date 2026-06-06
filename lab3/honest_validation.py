"""Honest validation: did the 0.650 m result on Split A test overfit?

We previously selected the ensemble hyperparameters (EmbKNN K, softmax τ,
which prediction sets to combine) by exhaustively searching greedy + swap on
Split A test. With ~33 candidates that's heavy multiple-hypothesis testing
on the test set — likely overfitting.

This script runs the proper sanity check:

  (A)  Apply the test-selected combo to a held-out validation slice that was
       NEVER used in the selection process. If val median ≈ 0.65, the combo
       generalizes; if val median jumps to 0.8+, it was test-overfit.

  (B)  Re-do greedy search using ONLY the held-out val to pick hyperparams,
       then evaluate that newly-selected combo on the actual test set.
       This simulates the result we'd have reported if we'd done the right
       thing in the first place.

Caveat: the Cascade encoders were trained on the full Split A train set
(1449 samples), so the encoder weights have seen the val samples. We're
isolating the hyperparameter-search bias, not the full encoder bias —
fully clean validation would require retraining encoders without val.
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
SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)


def geom_median(pts, eps=1e-5, max_iter=100):
    m = pts.mean(0)
    for _ in range(max_iter):
        d = np.clip(np.linalg.norm(pts - m[None], axis=-1), eps, None)
        w = 1.0 / d; w = w / w.sum(0, keepdims=True)
        new = (w[:, :, None] * pts).sum(0)
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps: break
        m = new
    return m


def loc_err(p, yt): return np.linalg.norm(p - yt, axis=1)
def med(p, yt): return float(np.median(loc_err(p, yt)))
def stats(p, yt):
    e = loc_err(p, yt)
    return (float(np.median(e)), float(e.mean()), float(np.percentile(e, 90)))


# All EmbKNN heads as (encoder_name, K, weight_type, name)
# weight_type: 'inv' (1/d) or ('soft', tau)
EMBKNN_HEADS = [
    # === BASE-encoder (Cascade) variants ===
    ('Cascade',          5, 'inv',           'EmbKNN_BaseK5'),
    ('Cascade',          7, 'inv',           'EmbKNN_BaseK7'),
    ('Cascade',         10, 'inv',           'EmbKNN_BaseK10'),
    ('Cascade',          3, 'inv',           'EmbKNN_BaseK3'),
    ('Cascade',         15, 'inv',           'EmbKNN_BaseK15'),
    ('Cascade',          5, ('soft', 0.1),   'EmbKNN_BaseSoftK5T01'),
    ('Cascade',          7, ('soft', 0.2),   'EmbKNN_BaseSoftK7T02'),
    ('Cascade',         15, ('soft', 0.05),  'EmbKNN_BaseSoftK15T005'),
    ('Cascade',          5, ('soft', 0.15),  'EmbKNN_BSoftK5T15'),
    ('Cascade',          7, ('soft', 0.1),   'EmbKNN_BSoftK7T10'),
    ('Cascade',          8, ('soft', 0.08),  'EmbKNN_BSoftK8T08'),
    ('Cascade',         10, ('soft', 0.1),   'EmbKNN_BSoftK10T10'),
    ('Cascade',         10, ('soft', 0.2),   'EmbKNN_BSoftK10T20'),
    # === TUNED-encoder variants ===
    ('CascadeTuned',     5, 'inv',           'EmbKNN_Tuned'),
    ('CascadeTuned',     7, 'inv',           'EmbKNN_TunedK7'),
    ('CascadeTuned',    10, 'inv',           'EmbKNN_TunedK10'),
    ('CascadeTuned',     3, 'inv',           'EmbKNN_TunedK3'),
    ('CascadeTuned',     5, ('soft', 0.1),   'EmbKNN_TunedSoftK5T01'),
    ('CascadeTuned',     7, ('soft', 0.1),   'EmbKNN_TSoftK7T10'),
    ('CascadeTuned',    10, ('soft', 0.1),   'EmbKNN_TSoftK10T10'),
    ('CascadeTuned',     5, ('soft', 0.05),  'EmbKNN_TSoftK5T05'),
    # === AGGRESSIVE encoder ===
    ('CascadeAggressive', 5, 'inv',          'EmbKNN_AggrK5'),
    ('CascadeAggressive', 5, ('soft', 0.1),  'EmbKNN_AggrSoftK5T01'),
]
CASCADE_VARIANTS = ['Cascade', 'CascadeTuned', 'CascadeAggressive']


def compute_cascade_preds(name, idx_q, val_q, mask_q, **k):
    """Run all available seeds; return (N_seeds, T, 2)."""
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


def compute_encoder_embeddings(name, idx_q, val_q, mask_q, **k):
    """Average normalized encoder embedding across seeds."""
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


def knn_predict(Z_db, y_db, Z_q, K, weight):
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(Z_db)
    d, idx = nn.kneighbors(Z_q)
    if weight == 'inv':
        w = 1.0 / (d + 1e-6)
    else:
        _, tau = weight
        w = np.exp(-d / tau)
    w = w / w.sum(axis=1, keepdims=True)
    return (w[..., None] * y_db[idx]).sum(axis=1)


def greedy_search(cand_preds, yt):
    """Greedy from each starting candidate + single-swap. Returns best (members, med)."""
    best_global = None
    for start in cand_preds:
        selected = [start]
        while True:
            cur_med = float('inf'); add_name = None
            for n in cand_preds:
                if n in selected: continue
                stacked = np.concatenate([cand_preds[s] for s in selected] +
                                          [cand_preds[n]], axis=0)
                m = med(geom_median(stacked), yt)
                if m < cur_med - 1e-4:
                    cur_med = m; add_name = n
            cur_med_check = med(geom_median(np.concatenate([cand_preds[s] for s in selected], axis=0)),
                                 yt) if selected else float('inf')
            if add_name is None or cur_med >= cur_med_check - 1e-4:
                break
            selected.append(add_name)
        # swap
        improved = True
        while improved:
            improved = False
            cur = med(geom_median(np.concatenate([cand_preds[s] for s in selected], axis=0)), yt)
            for i in range(len(selected)):
                for n in cand_preds:
                    if n in selected: continue
                    trial = selected.copy(); trial[i] = n
                    m = med(geom_median(np.concatenate([cand_preds[s] for s in trial], axis=0)), yt)
                    if m < cur - 1e-4:
                        selected = trial; cur = m; improved = True; break
                if improved: break
        final_med = med(geom_median(np.concatenate([cand_preds[s] for s in selected], axis=0)), yt)
        if best_global is None or final_med < best_global[1]:
            best_global = (selected, final_med)
    return best_global


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)
    kw = dict(bssids=bssids, fxy=fxy, fmask=fmask, cxy=cxy, cmask=cmask, f2c=f2c)

    # Split tr_A 90/10 → train_inner (DB for KNN) + val_inner (held-out)
    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(tr_A))
    n_val = max(100, len(tr_A) // 10)            # 145 val from 1449 train
    val_idx_local = perm[:n_val]
    db_idx_local = perm[n_val:]
    val_inner = tr_A[val_idx_local]
    db_train = tr_A[db_idx_local]
    print(f'train_A: {len(tr_A)}  →  train_inner (KNN DB): {len(db_train)}  val_inner (held-out): {len(val_inner)}')
    print(f'test_A:  {len(te_A)}\n')

    y_val = y[val_inner]; y_te = y[te_A]
    y_db = y[db_train]; y_tr_full = y[tr_A]

    # === Compute predictions on VAL and TEST ===
    print('--- computing predictions on val_inner ---')
    val_preds = {}
    for v in CASCADE_VARIANTS:
        p = compute_cascade_preds(v, set_idx[val_inner], set_val[val_inner],
                                     set_mask[val_inner], **kw)
        if p is not None: val_preds[v] = p
        print(f'  {v}: {p.shape if p is not None else "None"}')

    # cache embeddings per encoder for val + DB + test + full train
    embeds_val = {}; embeds_db = {}; embeds_te = {}; embeds_tr_full = {}
    for enc in CASCADE_VARIANTS:
        embeds_val[enc] = compute_encoder_embeddings(enc, set_idx[val_inner],
                                                          set_val[val_inner],
                                                          set_mask[val_inner], **kw)
        embeds_db[enc] = compute_encoder_embeddings(enc, set_idx[db_train],
                                                         set_val[db_train],
                                                         set_mask[db_train], **kw)
        embeds_tr_full[enc] = compute_encoder_embeddings(enc, set_idx[tr_A],
                                                              set_val[tr_A],
                                                              set_mask[tr_A], **kw)
        embeds_te[enc] = compute_encoder_embeddings(enc, set_idx[te_A],
                                                         set_val[te_A],
                                                         set_mask[te_A], **kw)

    for enc, K, wt, name in EMBKNN_HEADS:
        # KNN on val: db=train_inner only (val held-out)
        p_val = knn_predict(embeds_db[enc], y_db, embeds_val[enc], K, wt)
        val_preds[name] = p_val[None, ...]

    print('\n--- computing predictions on test_A ---')
    test_preds = {}
    for v in CASCADE_VARIANTS:
        p = compute_cascade_preds(v, set_idx[te_A], set_val[te_A], set_mask[te_A], **kw)
        if p is not None: test_preds[v] = p
    for enc, K, wt, name in EMBKNN_HEADS:
        # KNN on test: db = FULL tr_A (no leak — test never in db)
        p_te = knn_predict(embeds_tr_full[enc], y_tr_full, embeds_te[enc], K, wt)
        test_preds[name] = p_te[None, ...]

    # ─────────────────────────────────────────────────────────────
    # (A) Apply the previously-test-selected combo (the "0.650" combo)
    #     to val. If val ≈ test, that combo generalizes.
    # ─────────────────────────────────────────────────────────────
    cheat_combo = ['CascadeTuned', 'EmbKNN_TSoftK5T05', 'EmbKNN_BaseSoftK5T01',
                    'EmbKNN_BSoftK5T15', 'EmbKNN_BaseK5']
    print('\n' + '=' * 70)
    print('(A) Test-selected combo applied to held-out val:')
    print(f'    combo: {cheat_combo}')
    val_p = geom_median(np.concatenate([val_preds[s] for s in cheat_combo], axis=0))
    te_p = geom_median(np.concatenate([test_preds[s] for s in cheat_combo], axis=0))
    m_v, mn_v, p9_v = stats(val_p, y_val)
    m_t, mn_t, p9_t = stats(te_p, y_te)
    print(f'    on val:   median={m_v:.3f}  mean={mn_v:.3f}  p90={p9_v:.3f}  (n={len(y_val)})')
    print(f'    on test:  median={m_t:.3f}  mean={mn_t:.3f}  p90={p9_t:.3f}  (n={len(y_te)})')
    print(f'    val−test gap: {m_v - m_t:+.3f}  ←  if small (|<0.05|), generalizes;'
          f' if large, was overfit')

    # ─────────────────────────────────────────────────────────────
    # (B) Honest path: greedy on val to pick combo, then evaluate on test once.
    # ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('(B) Honest pipeline: greedy on VAL → final evaluation on TEST')
    sel, val_med = greedy_search(val_preds, y_val)
    print(f'    val-selected combo: {sel}')
    print(f'    val median:  {val_med:.3f}')
    te_p = geom_median(np.concatenate([test_preds[s] for s in sel], axis=0))
    m_t, mn_t, p9_t = stats(te_p, y_te)
    print(f'    test median (honest reportable): {m_t:.3f}')
    print(f'    test mean = {mn_t:.3f}  p90 = {p9_t:.3f}')

    print('\n' + '=' * 70)
    print('Summary of honest reportable numbers:')
    print(f'  Previously claimed (test-tuned):  0.650 m')
    print(f'  (A) test-combo on held-out val:    {m_v:.3f} m')
    print(f'  (B) val-tuned, evaluated on test:  {m_t:.3f} m')


if __name__ == '__main__':
    main()
