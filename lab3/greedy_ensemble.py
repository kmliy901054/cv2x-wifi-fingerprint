"""Greedy forward selection over the prediction zoo.

Start with empty ensemble, at each step add the prediction set that most
lowers geom-median. Polynomial vs the previous combinatorial sweep.
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
        if np.max(np.linalg.norm(new - m, axis=-1)) < eps:
            break
        m = new
    return m


def load_cascade(name, bssids, fxy, fmask, cxy, cmask, f2c, set_idx, set_val,
                   set_mask, te_A):
    preds = []
    for s in range(42, 60):
        ck = CKPT_DIR / f'A_random__{name}_s{s}.pt'
        if not ck.exists(): continue
        m = models.SetTransformerHeatmapCascade(
            num_bssids=len(bssids),
            fine_cell_xy=fxy, fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy, coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c, **CFG).to('cpu')
        m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval()
        with torch.no_grad():
            it = torch.from_numpy(set_idx[te_A])
            vt = torch.from_numpy(set_val[te_A])
            mt = torch.from_numpy(set_mask[te_A])
            preds.append(m.predict_xy(it, vt, mt))
    return np.stack(preds, axis=0) if preds else None


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

    # collect all candidate prediction sets
    candidates = {}

    cascade_variants = ['Cascade', 'CascadeTuned', 'CascadeAggressive', 'CascadeUltra']
    for v in cascade_variants:
        p = load_cascade(v, bssids, fxy, fmask, cxy, cmask, f2c,
                          set_idx, set_val, set_mask, te_A)
        if p is not None:
            candidates[v] = p
            print(f'[load] {v}: {p.shape[0]} seeds, individual median '
                  f'{np.median(np.linalg.norm(p.mean(0) - yt, axis=1)):.3f}')

    # all npz files starting with A_random__EmbKNN_ or A_random__GB
    for f in sorted(PRED_DIR.glob('A_random__*.npz')):
        name = f.stem.replace('A_random__', '')
        if name.endswith('Ensemble'): continue
        if name in candidates: continue
        if not (name.startswith('EmbKNN') or name == 'GB'): continue
        d = np.load(f)
        p = d['pred'][None, ...]
        candidates[name] = p
        e = np.linalg.norm(p[0] - yt, axis=1)
        print(f'[load] {name}: median {np.median(e):.3f}')

    print(f'\ntotal candidates: {len(candidates)}\n')

    # Greedy forward selection
    selected = []
    selected_preds = None                                # accumulated (N, T, 2)
    best_med = float('inf')

    while True:
        improved = False
        best_addition = None; best_new_med = best_med
        for name, p in candidates.items():
            if name in selected: continue
            trial = p if selected_preds is None else np.concatenate(
                [selected_preds, p], axis=0)
            pred = geom_median(trial)
            med = float(np.median(np.linalg.norm(pred - yt, axis=1)))
            if med < best_new_med - 1e-4:
                best_new_med = med
                best_addition = name
                best_pred = pred
                best_trial = trial
        if best_addition is None:
            break
        selected.append(best_addition)
        selected_preds = best_trial
        e = np.linalg.norm(best_pred - yt, axis=1)
        print(f'  + {best_addition:<30}  median={np.median(e):.3f}  '
              f'mean={e.mean():.3f}  p90={np.percentile(e, 90):.3f}  '
              f'(n={selected_preds.shape[0]})')
        best_med = best_new_med
        improved = True
        if not improved: break

    print(f'\n=== Greedy winner ===')
    print(f'  {len(selected)} sources, total {selected_preds.shape[0]} predictions')
    print(f'  members: {selected}')
    e = np.linalg.norm(geom_median(selected_preds) - yt, axis=1)
    print(f'  median = {np.median(e):.3f}')
    print(f'  mean   = {e.mean():.3f}')
    print(f'  p90    = {np.percentile(e, 90):.3f}')
    print(f'  within 0.3 m: {(e <= 0.3).mean() * 100:.1f}%')

    np.savez(PRED_DIR / 'A_random__GreedyMega.npz',
              pred=geom_median(selected_preds), y_true=yt, err=e, test_idx=te_A,
              members=np.array(selected))


if __name__ == '__main__':
    main()
