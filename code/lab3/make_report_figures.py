"""Generate report figures from saved prediction npz files.

Output PNGs into outputs/figures/:
  1. error_cdf.png            — CDF of localization error, all major models
  2. error_boxplot.png        — per-model error distribution (log scale)
  3. cascade_scatter.png      — Cascade prediction scatter on floor plan
  4. cascade_worst10.png      — 10 worst predictions visualized with arrows
  5. ladder_bar.png           — bar chart of median error across all models
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PMingLiU', 'SimHei',
                                     'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[2]
LAB3 = Path(__file__).parent
PRED_DIR = LAB3 / 'outputs' / 'predictions'
FIG_DIR = LAB3 / 'outputs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pred(name):
    f = PRED_DIR / f'A_random__{name}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    return d['pred'], d['y_true'], np.linalg.norm(d['pred'] - d['y_true'], axis=1)


def load_map():
    yaml_path = ROOT / 'map' / 'psquare.yaml'
    with open(yaml_path) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(yaml_path.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution']


# Ordered: KNN baseline → climbing → final winner → failures
MODELS = [
    ('KNN_k5',                   'KNN k=5',                 '#999999', '-', 'baseline'),
    ('BestComboEnsemble',        'Big+synth ×5-ens (MDN)',  '#1f77b4', '-', 'pre-Heatmap'),
    ('CMixupComboEnsemble',      'C-Mixup ×5-ens',          '#d62728', '--', 'failed'),
    ('HeatmapEnsemble',          'Heatmap+mask ×5-ens',     '#ff7f0e', '-', 'climbing'),
    ('CascadeEnsemble',          '[winner] Cascade ×5-ens',        '#2ca02c', '-', 'WINNER'),
    ('HeatmapMapEnsemble',       'CNN xattn ×5-ens',        '#9467bd', '--', 'failed'),
    ('Cascade3Ensemble',         '3-Cascade ×10-ens',       '#8c564b', '--', 'failed'),
    ('DiffusionEnsemble',        'Diffusion ×5-ens',        '#e377c2', '--', 'failed'),
]


def fig_error_cdf():
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, label, color, ls, _ in MODELS:
        out = load_pred(name)
        if out is None:
            print(f'  skip {name}: missing')
            continue
        _, _, err = out
        x = np.sort(err)
        y = np.arange(1, len(x) + 1) / len(x)
        med = np.median(err)
        ax.plot(x, y, label=f'{label}  (med={med:.3f})', color=color,
                 linestyle=ls, linewidth=2)
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.6,
               label='AMCL noise floor (0.3 m)')
    ax.set_xlim(0, 5)
    ax.set_xlabel('Localization Error (m)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Split A Test Error CDF — All Major Models')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'error_cdf.png', dpi=150)
    plt.close(fig)
    print('  ✓ error_cdf.png')


def fig_error_boxplot():
    data = []; labels = []; colors = []
    for name, label, color, _, _ in MODELS:
        out = load_pred(name)
        if out is None: continue
        _, _, err = out
        data.append(err)
        labels.append(label)
        colors.append(color)
    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=labels, vert=False, patch_artist=True,
                     showfliers=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axvline(0.793, color='green', linestyle='--', alpha=0.7,
               label='Cascade median 0.793 m')
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.6,
               label='AMCL noise floor')
    ax.set_xlabel('Localization Error (m)')
    ax.set_xlim(0, 6)
    ax.set_title('Per-model Error Distribution (Split A, log-friendly view)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'error_boxplot.png', dpi=150)
    plt.close(fig)
    print('  ✓ error_boxplot.png')


def fig_cascade_scatter():
    """Two-panel scatter: TRUE positions colored by error (where the
    model fails geographically), + predicted positions for comparison."""
    pred, y_true, err = load_pred('CascadeEnsemble')
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]
    # crop to actual lab area for better visibility
    xlim = (-7, 12)
    ylim = (-3, 14)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, xy, title in [
        (axes[0], y_true, 'TRUE positions colored by error\n'
                          '(shows WHERE the model fails geographically)'),
        (axes[1], pred,   'PREDICTED positions colored by error\n'
                          '(shows where the model THINKS test samples are)'),
    ]:
        ax.imshow(pgm, cmap='gray', origin='lower', extent=extent, alpha=0.5)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=err, cmap='inferno',
                         s=22, edgecolors='black', linewidths=0.3,
                         vmin=0, vmax=3.0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('Localization Error (m)')
    plt.suptitle(f'Cascade  —  363 test samples, median = '
                  f'{np.median(err):.3f} m, mean = {err.mean():.3f} m '
                  f'(27% within AMCL 0.3 m noise floor)', fontsize=12)
    fig.savefig(FIG_DIR / 'cascade_scatter.png', dpi=150,
                  bbox_inches='tight')
    plt.close(fig)
    print('  ✓ cascade_scatter.png')


def fig_data_coverage():
    """Train + test true positions on the floor plan to show data coverage."""
    import data
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]
    xlim = (-7, 12); ylim = (-3, 14)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(pgm, cmap='gray', origin='lower', extent=extent, alpha=0.5)
    ax.scatter(y[tr_A, 0], y[tr_A, 1], s=10, alpha=0.4, c='steelblue',
               label=f'train ({len(tr_A)})')
    ax.scatter(y[te_A, 0], y[te_A, 1], s=18, alpha=0.7, c='crimson',
               edgecolors='black', linewidths=0.3,
               label=f'test ({len(te_A)})')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Split A data coverage on floor plan\n'
                 f'Samples concentrated in lower lab; upper area '
                 f'(y > 10 m) has almost no data')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'data_coverage.png', dpi=150)
    plt.close(fig)
    print('  ✓ data_coverage.png')


def fig_cascade_worst10():
    pred, y_true, err = load_pred('CascadeEnsemble')
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    worst_i = np.argsort(err)[-10:][::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(pgm, cmap='gray', origin='lower', extent=extent, alpha=0.5)
    # all test points light grey
    ax.scatter(y_true[:, 0], y_true[:, 1], c='lightblue', s=8, alpha=0.5,
               label='all test ground truth')
    # plot top-10 worst with arrows from true → pred
    for rank, i in enumerate(worst_i, 1):
        tx, ty = y_true[i]
        px, py = pred[i]
        ax.plot([tx, px], [ty, py], 'r-', alpha=0.6, linewidth=1.5)
        ax.scatter(tx, ty, c='green', s=80, marker='o', zorder=5,
                    edgecolors='black', linewidths=1)
        ax.scatter(px, py, c='red', s=80, marker='x', zorder=5, linewidths=2)
        ax.annotate(f'#{rank}\n{err[i]:.2f}m', (tx, ty), xytext=(5, 5),
                     textcoords='offset points', fontsize=8,
                     color='black', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Top-10 Worst Cascade Predictions\n'
                 'green ○ = true, red × = predicted, line = error vector')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'cascade_worst10.png', dpi=150)
    plt.close(fig)
    print('  ✓ cascade_worst10.png')


def fig_ladder_bar():
    # Hand-curated ladder data (median errors from our experiments)
    rows = [
        ('KNN k=5',              1.568, 'baseline',  '#999999'),
        ('MLP',                   1.34,  'baseline',  '#bbbbbb'),
        ('MaskedMDN',             1.20,  'baseline',  '#cccccc'),
        ('Set Transformer MDN',  1.093, 'climbing',  '#1f77b4'),
        ('Big × 5-ens (MDN)',    1.083, 'climbing',  '#1f77b4'),
        ('+ GP synth (single)',  0.906, 'breakthrough', '#ff7f0e'),
        ('+ GP synth × 5-ens',   0.889, 'breakthrough', '#ff7f0e'),
        ('C-Mixup × 5-ens',      0.942, 'failed',    '#d62728'),
        ('Heatmap × 5-ens',      0.836, 'climbing',  '#1f77b4'),
        ('Cascade × 5-ens',      0.793, 'winner',    '#2ca02c'),
        ('CNN xattn × 5-ens',    0.907, 'failed',    '#d62728'),
        ('A+B combo × 4-ens',    0.886, 'failed',    '#d62728'),
        ('3-Cascade × 10-ens',   0.800, 'failed',    '#d62728'),
        ('Diffusion × 5-ens',    1.821, 'failed',    '#d62728'),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(rows))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor='black', alpha=0.8)
    # annotate
    for bar, v in zip(bars, vals):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.793, color='green', linestyle='--', alpha=0.7,
               label='Winner: Cascade 0.793 m')
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.6,
               label='AMCL noise floor 0.3 m')
    ax.set_xlabel('Median Location Error (m)')
    ax.set_title('Split A Median Error — Full Experimental Ladder')
    ax.set_xlim(0, 2.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ladder_bar.png', dpi=150)
    plt.close(fig)
    print('  ✓ ladder_bar.png')


def main():
    print(f'[fig] writing to {FIG_DIR}')
    fig_error_cdf()
    fig_error_boxplot()
    fig_cascade_scatter()
    fig_cascade_worst10()
    fig_ladder_bar()
    fig_data_coverage()
    print('Done.')


if __name__ == '__main__':
    main()
