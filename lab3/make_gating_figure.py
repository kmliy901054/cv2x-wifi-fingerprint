"""The thesis figure: why the coarse→fine cascade wins.

Picks the Split-A test sample where the coarse gate rescues the prediction the
most (fine head alone is multi-modal / ambiguous; the coarse gate collapses it
to the right region). Renders three heatmap panels over the floor plan:

  1. Fine head alone        softmax(fine) ⊙ free_mask         (multi-modal, wrong)
  2. Coarse head            softmax(coarse) ⊙ free_mask        (the gate)
  3. Gated (fine × coarse)  the cascade output                 (single peak, right)

Output: outputs/figures/gating_before_after.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PMingLiU', 'SimHei',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import data
import models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
CKPT = HERE / 'outputs' / 'checkpoints' / 'A_random__Cascade_s42.pt'
FIG_DIR = HERE / 'outputs' / 'figures'
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)


def load_map():
    yp = ROOT / 'map' / 'psquare.yaml'
    m = yaml.safe_load(open(yp))
    pgm = np.array(Image.open(yp.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution']


def overlay(ax, pgm, extent):
    ax.imshow(np.flipud(pgm), cmap='gray', origin='lower', extent=extent,
              alpha=0.45, interpolation='nearest')


def main():
    device = torch.device('cpu')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    _, te = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=50)

    fine_xy, Gw, Gh = data.build_heatmap_grid(cell_size=0.4)
    fine_mask = data.build_free_mask(fine_xy)
    coarse_xy, Cw, Ch = data.build_heatmap_grid(cell_size=1.6)
    coarse_mask = data.build_free_mask(coarse_xy)
    f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)

    model = models.SetTransformerHeatmapCascade(
        num_bssids=len(bssids),
        fine_cell_xy=fine_xy, fine_free_mask=fine_mask.astype(np.float32),
        coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
        fine_to_coarse=f2c, **CFG).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    it = torch.from_numpy(set_idx[te])
    vt = torch.from_numpy(set_val[te])
    mt = torch.from_numpy(set_mask[te])
    with torch.no_grad():
        fine_l, coarse_l = model(it, vt, mt)
        fine_p = F.softmax(fine_l + model.log_fine_mask.unsqueeze(0), dim=-1)
        coarse_p = F.softmax(coarse_l + model.log_coarse_mask.unsqueeze(0), dim=-1)
        gated = model._gated_fine_prob(fine_l, coarse_l)
        xy_fine = (fine_p @ model.fine_xy).numpy()
        xy_gated = (gated @ model.fine_xy).numpy()
    fxy = model.fine_xy.numpy()
    true = y[te]
    err_fine = np.linalg.norm(xy_fine - true, axis=1)
    err_gated = np.linalg.norm(xy_gated - true, axis=1)

    # multi-modality of fine_p: 1 - (top1 mass / top2 mass) heuristic; use entropy
    fp = fine_p.numpy()
    # second mode separation: distance between top-1 cell and the farthest
    # high-mass cell
    score = np.zeros(len(te))
    for i in range(len(te)):
        p = fp[i]
        top = np.argsort(p)[::-1][:8]
        if p[top[0]] <= 0:
            continue
        c0 = fxy[top[0]]
        # mass-weighted spread of the top cells = how spread out the fine belief is
        w = p[top]; w = w / w.sum()
        spread = np.sqrt((w * np.sum((fxy[top] - c0) ** 2, axis=1)).sum())
        score[i] = spread

    # pick: fine ambiguous (big spread) AND gating clearly helps
    improvement = err_fine - err_gated
    cand = np.where((score > np.percentile(score, 70)) &
                    (err_gated < 1.0) & (improvement > 1.0))[0]
    if len(cand) == 0:
        cand = np.argsort(improvement)[::-1][:5]
    i = cand[np.argmax(improvement[cand])]

    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]
    xlim = (-7, 12); ylim = (-3, 14)

    def heat_panel(ax, prob, cell_xy, gw, gh, cell, title, mark_xy, mark_err):
        overlay(ax, pgm, extent)
        # build grid image
        grid = prob.reshape(gh, gw)
        x0 = cell_xy[:, 0].min() - cell / 2
        y0 = cell_xy[:, 1].min() - cell / 2
        ext = [x0, x0 + gw * cell, y0, y0 + gh * cell]
        m = np.ma.masked_where(grid <= grid.max() * 0.02, grid)
        ax.imshow(m, origin='lower', extent=ext, cmap='turbo', alpha=0.82,
                  interpolation='nearest', aspect='auto')
        ax.scatter(*true[i], marker='*', s=320, c='gold',
                   edgecolors='black', linewidths=1.2, zorder=6, label='true')
        ax.scatter(*mark_xy, marker='X', s=130, c='white',
                   edgecolors='black', linewidths=1.5, zorder=6,
                   label=f'predicted ({mark_err:.2f} m)')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x (m)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    heat_panel(axes[0], fp[i], fxy, Gw, Gh, 0.4,
               f'① Fine head alone\nmulti-modal, expected pos is off',
               xy_fine[i], err_fine[i])
    heat_panel(axes[1], coarse_p.numpy()[i], coarse_xy, Cw, Ch, 1.6,
               f'② Coarse head (the gate)\n10×9 grid picks the right region',
               xy_gated[i], err_gated[i])
    heat_panel(axes[2], gated.numpy()[i], fxy, Gw, Gh, 0.4,
               f'③ Gated = fine × coarse\nsingle peak, correct',
               xy_gated[i], err_gated[i])
    axes[0].set_ylabel('y (m)')
    plt.suptitle(
        f'Why the cascade wins: the coarse gate removes fine-grid ambiguity  '
        f'(this sample: {err_fine[i]:.2f} m → {err_gated[i]:.2f} m)',
        fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / 'gating_before_after.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out.name}   (sample {i}: fine {err_fine[i]:.2f} m, '
          f'gated {err_gated[i]:.2f} m)')


if __name__ == '__main__':
    main()
