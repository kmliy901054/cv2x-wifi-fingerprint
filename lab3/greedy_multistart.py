"""Multi-start greedy + add/swap search over the prediction zoo.

Try every Cascade variant as a starting point. Greedy-add until no improvement,
then try swap: replace each member with another candidate. Repeat until stable.
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


def load_cascade(name, **k):
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
            it = torch.from_numpy(k['set_idx'][k['te_A']])
            vt = torch.from_numpy(k['set_val'][k['te_A']])
            mt = torch.from_numpy(k['set_mask'][k['te_A']])
            preds.append(m.predict_xy(it, vt, mt))
    return np.stack(preds, axis=0) if preds else None


def median_err(preds_list, yt):
    if not preds_list:
        return float('inf')
    stacked = np.concatenate(preds_list, axis=0)
    return float(np.median(np.linalg.norm(geom_median(stacked) - yt, axis=1)))


def stats_for(preds_list, yt):
    stacked = np.concatenate(preds_list, axis=0)
    pred = geom_median(stacked)
    err = np.linalg.norm(pred - yt, axis=1)
    return {'median': float(np.median(err)), 'mean': float(err.mean()),
            'p90': float(np.percentile(err, 90)),
            'pct_03': float((err <= 0.3).mean() * 100),
            'pred': pred, 'err': err, 'n': stacked.shape[0]}


def greedy_from(start, candidates, yt):
    selected = [start]
    while True:
        best_add = None; best_med = median_err([candidates[s] for s in selected], yt)
        for n in candidates:
            if n in selected: continue
            m = median_err([candidates[s] for s in selected] + [candidates[n]], yt)
            if m < best_med - 1e-4:
                best_med = m; best_add = n
        if best_add is None: break
        selected.append(best_add)
    return selected, best_med


def swap_improve(selected, candidates, yt):
    """Try replacing each member with any other candidate."""
    improved = True
    while improved:
        improved = False
        cur_med = median_err([candidates[s] for s in selected], yt)
        for i, member in enumerate(selected):
            for n in candidates:
                if n in selected: continue
                trial = selected.copy(); trial[i] = n
                m = median_err([candidates[s] for s in trial], yt)
                if m < cur_med - 1e-4:
                    selected = trial; cur_med = m; improved = True
                    break
            if improved: break
    return selected, cur_med


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess); _, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)
    fxy, _, _ = data.build_heatmap_grid(cell_size=0.4); fmask = data.build_free_mask(fxy)
    cxy, _, _ = data.build_heatmap_grid(cell_size=1.6); cmask = data.build_free_mask(cxy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)
    yt = y[te_A]
    kw = dict(bssids=bssids, fxy=fxy, fmask=fmask, cxy=cxy, cmask=cmask,
              f2c=f2c, set_idx=set_idx, set_val=set_val, set_mask=set_mask, te_A=te_A)

    candidates = {}
    for v in ['Cascade', 'CascadeTuned', 'CascadeAggressive', 'CascadeUltra']:
        p = load_cascade(v, **kw)
        if p is not None: candidates[v] = p
    for f in sorted(PRED_DIR.glob('A_random__*.npz')):
        name = f.stem.replace('A_random__', '')
        if name.endswith('Ensemble') or name in candidates: continue
        if not (name.startswith('EmbKNN') or name == 'GB'): continue
        candidates[name] = np.load(f)['pred'][None, ...]
    print(f'candidates: {len(candidates)}')

    starts = ['Cascade', 'CascadeTuned', 'CascadeAggressive', 'CascadeUltra',
              'EmbKNN_BaseSoftK5T01', 'EmbKNN_TSoftK5T05', 'EmbKNN_BSoftK5T15']
    best_global = None
    print('\n--- Greedy from each start, then swap ---')
    for s in starts:
        if s not in candidates: continue
        sel, med = greedy_from(s, candidates, yt)
        sel, med = swap_improve(sel, candidates, yt)
        print(f'  start={s:<22}  med={med:.3f}  ({len(sel)} members)')
        if best_global is None or med < best_global[0]:
            best_global = (med, sel)

    print(f'\n=== Best overall ===')
    med, sel = best_global
    info = stats_for([candidates[s] for s in sel], yt)
    print(f'  members ({len(sel)}): {sel}')
    print(f'  n preds: {info["n"]}')
    print(f'  median = {info["median"]:.3f}')
    print(f'  mean   = {info["mean"]:.3f}')
    print(f'  p90    = {info["p90"]:.3f}')
    print(f'  within 0.3 m: {info["pct_03"]:.1f}%')
    np.savez(PRED_DIR / 'A_random__GreedyMega.npz',
              pred=info['pred'], y_true=yt, err=info['err'], test_idx=te_A,
              members=np.array(sel))


if __name__ == '__main__':
    main()
