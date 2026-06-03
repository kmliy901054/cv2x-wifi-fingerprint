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


def clean_map_overlay(ax, pgm, extent, alpha=0.55):
    """Render the SLAM floor plan as a backdrop, properly oriented.

    PGM stores row 0 at the top of the image (north in world); without flipud
    a plain imshow puts it upside down vs the data trajectory. We pre-flip and
    keep origin='lower' so the y axis points up like ROS map frame."""
    ax.imshow(np.flipud(pgm), cmap='gray', origin='lower', extent=extent,
              alpha=alpha, interpolation='nearest')


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
        clean_map_overlay(ax, pgm, extent)
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
    clean_map_overlay(ax, pgm, extent)
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


def fig_cascade_random100():
    """100 random test samples (excluding extreme outliers > 2.5 m), error
    vectors true → pred. Line color = error magnitude on a green→red palette,
    line width scales with error so long outliers stand out clearly."""
    pred, y_true, err = load_pred('CascadeEnsemble')
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    # Exclude the worst 5% so the figure isn't dominated by a few cross-map lines
    cutoff = np.percentile(err, 95)
    eligible = np.where(err <= cutoff)[0]
    rng = np.random.default_rng(42)
    n = min(100, len(eligible))
    sel = rng.choice(eligible, size=n, replace=False)

    fig, ax = plt.subplots(figsize=(11, 10))
    clean_map_overlay(ax, pgm, extent)
    ax.scatter(y_true[:, 0], y_true[:, 1], c='lightblue', s=6, alpha=0.35,
               label=f'all test ({len(err)})')
    # Truncated 'plasma' (skip both pale ends) — dark violet → magenta → dark orange.
    # No pale colors anywhere, every line stays readable on white background.
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap('plasma')
    cmap = LinearSegmentedColormap.from_list(
        'plasma_mid', [base(t) for t in np.linspace(0.78, 0.20, 64)])
    vmin, vmax = 0.0, 2.5
    for i in sel:
        tx, ty = y_true[i]
        px, py = pred[i]
        c = cmap(min(1.0, err[i] / vmax))
        # line width scales with error: 1.0 → 2.8 px
        lw = 1.0 + 1.8 * min(1.0, err[i] / vmax)
        ax.plot([tx, px], [ty, py], color=c, alpha=0.85, linewidth=lw,
                zorder=3, solid_capstyle='round')
        ax.scatter(tx, ty, c='#1a7a1a', s=28, marker='o', zorder=4,
                   edgecolors='black', linewidths=0.5)
        ax.scatter(px, py, c='#c0392b', s=28, marker='x', zorder=5,
                   linewidths=1.6)
    sm = plt.cm.ScalarMappable(cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('Error (m)')
    sel_err = err[sel]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Random 100 Cascade Predictions  (worst 5% excluded)\n'
                 f'green ● = true, red × = predicted, line = error vector\n'
                 f'sample median {np.median(sel_err):.2f} m, '
                 f'mean {sel_err.mean():.2f} m  ({np.median(err):.3f} / '
                 f'{err.mean():.3f} m on all 363)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'cascade_random100.png', dpi=150)
    plt.close(fig)
    print('  ✓ cascade_random100.png')


def fig_cascade_worst10():
    pred, y_true, err = load_pred('CascadeEnsemble')
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    worst_i = np.argsort(err)[-10:][::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    clean_map_overlay(ax, pgm, extent)
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
        ('MLP',                   1.302, 'baseline',  '#bbbbbb'),
        ('MaskedMDN',             1.371, 'baseline',  '#cccccc'),
        ('Set Transformer MDN',  1.093, 'climbing',  '#1f77b4'),
        ('Big × 5-ens (MDN)',    1.083, 'climbing',  '#1f77b4'),
        ('+ GP synth (single)',  0.906, 'breakthrough', '#ff7f0e'),
        ('+ GP synth × 5-ens',   0.889, 'breakthrough', '#ff7f0e'),
        ('C-Mixup × 5-ens',      0.942, 'failed',    '#d62728'),
        ('Heatmap × 5-ens',      0.883, 'climbing',  '#1f77b4'),
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


def fig_region_reliability():
    """Bin the floor into cells, color each by the mean error of test samples
    whose TRUE position lands there. Shows geographic reliability: where the
    model is trustworthy (dense, low error) vs unreliable (sparse upper area)."""
    pred, y_true, err = load_pred('CascadeEnsemble')
    pgm, ox, oy, res = load_map()
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]

    cell = 1.0           # 1 m reliability bins
    xlim = (-7, 12); ylim = (-3, 14)
    xedges = np.arange(xlim[0], xlim[1] + cell, cell)
    yedges = np.arange(ylim[0], ylim[1] + cell, cell)
    # mean error + count per bin
    sum_e, _, _ = np.histogram2d(y_true[:, 0], y_true[:, 1], bins=[xedges, yedges],
                                 weights=err)
    cnt, _, _ = np.histogram2d(y_true[:, 0], y_true[:, 1], bins=[xedges, yedges])
    mean_e = np.full_like(sum_e, np.nan)
    nz = cnt > 0
    mean_e[nz] = sum_e[nz] / cnt[nz]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    # Panel 1: reliability (mean error per cell)
    ax = axes[0]
    clean_map_overlay(ax, pgm, extent, alpha=0.35)
    masked = np.ma.masked_invalid(mean_e.T)   # transpose: histogram2d is (x,y)
    pcm = ax.pcolormesh(xedges, yedges, masked, cmap='RdYlGn_r',
                        vmin=0, vmax=2.0, alpha=0.78, shading='flat',
                        edgecolors='white', linewidth=0.3)
    cb = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Mean error in cell (m)')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Geographic reliability\n(green = trustworthy, red = unreliable)')
    # Panel 2: sample density per cell
    ax = axes[1]
    clean_map_overlay(ax, pgm, extent, alpha=0.35)
    masked_c = np.ma.masked_where(cnt.T == 0, cnt.T)
    pcm2 = ax.pcolormesh(xedges, yedges, masked_c, cmap='viridis',
                         alpha=0.78, shading='flat',
                         edgecolors='white', linewidth=0.3)
    cb2 = plt.colorbar(pcm2, ax=ax, shrink=0.7, pad=0.02)
    cb2.set_label('# test samples in cell')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Test sample density\n(reliability tracks data coverage)')
    plt.suptitle('Where can you trust the model? Error is low where data is dense, '
                 'high in the sparse upper region.', fontsize=12)
    fig.savefig(FIG_DIR / 'region_reliability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ region_reliability.png')


def fig_error_vs_aps_and_drift():
    """Two panels: (1) error vs number of matched known APs — justifies the
    LOW CONFIDENCE guard; (2) per-BSSID morning vs evening mean RSSI — shows
    the day/night drift the model has to be robust to."""
    import data
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    vocab = {b.upper() for b in bssids}

    pred, y_true, err = load_pred('CascadeEnsemble')
    d = np.load(PRED_DIR / 'A_random__CascadeEnsemble.npz')
    test_idx = d['test_idx']

    # matched AP count per test sample
    n_matched = np.array([
        sum(1 for b in records[i]['aps'] if b.upper() in vocab)
        for i in test_idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel 1: error vs matched APs (binned mean + scatter)
    ax = axes[0]
    ax.scatter(n_matched, err, s=18, alpha=0.35, c='#4477aa',
               edgecolors='none', label='test samples')
    # binned median
    bins = np.arange(n_matched.min(), n_matched.max() + 2)
    bmid, bmed = [], []
    for lo in bins[:-1]:
        m = (n_matched >= lo) & (n_matched < lo + 1)
        if m.sum() >= 3:
            bmid.append(lo); bmed.append(np.median(err[m]))
    ax.plot(bmid, bmed, 'o-', color='#cc3311', linewidth=2.2, markersize=6,
            label='median per AP-count', zorder=5)
    ax.axhline(0.793, color='green', linestyle='--', alpha=0.6,
               label='overall median 0.793 m')
    ax.set_xlabel('# matched known APs in scan')
    ax.set_ylabel('Localization error (m)')
    ax.set_ylim(0, 5)
    ax.set_title('More matched APs → lower error\n(justifies the LOW-CONFIDENCE guard)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: morning vs evening mean RSSI per BSSID
    ax = axes[1]
    morn, even = {}, {}
    for r in records:
        tgt = morn if r['session'] == 'morning' else even
        for b, rssi in r['aps'].items():
            tgt.setdefault(b.upper(), []).append(rssi)
    common = [b for b in vocab if b in morn and b in even
              and len(morn[b]) >= 20 and len(even[b]) >= 20]
    mx = np.array([np.mean(morn[b]) for b in common])
    ex = np.array([np.mean(even[b]) for b in common])
    drift = ex - mx
    sc = ax.scatter(mx, ex, c=np.abs(drift), cmap='plasma', s=55,
                    edgecolors='black', linewidths=0.5, vmin=0, vmax=6)
    lim = [-95, -25]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='no drift (y = x)')
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('|drift| (dBm)')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Morning mean RSSI (dBm)')
    ax.set_ylabel('Evening mean RSSI (dBm)')
    ax.set_title(f'Morning vs evening RSSI drift\n'
                 f'{len(common)} APs, median |drift| = '
                 f'{np.median(np.abs(drift)):.1f} dBm')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'error_vs_aps_drift.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ error_vs_aps_drift.png')


def fig_synth_ablation():
    """Controlled synthetic-data ablation. Same Set Transformer architecture,
    same training; only difference = +5000 GP-kriging synthetic scans.
    Panel 1: error CDF on Split A (synth helps). Panel 2: median across both
    Split A (random) and Split C (morning→evening) — the honest result that
    synth fills spatial gaps but carries morning bias on cross-session transfer."""
    def load(f):
        d = np.load(PRED_DIR / f)
        return d['err'] if 'err' in d else np.linalg.norm(
            d['pred'] - d['y_true'], axis=1)

    A_real = load('A_random__SetTransformerBig_RealOnly.npz')
    A_syn  = load('A_random__SetTransformerBig_PlusSyntheticA.npz')
    C_real = load('C_morning_to_evening__SetTransformerBig_RealOnly.npz')
    C_syn  = load('C_morning_to_evening__SetTransformerBig_PlusSynthetic.npz')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1 — CDF on Split A
    ax = axes[0]
    for err, label, color in [
        (A_real, f'real only  (med {np.median(A_real):.3f})', '#888888'),
        (A_syn,  f'real + 5000 synth  (med {np.median(A_syn):.3f})', '#2ca02c'),
    ]:
        x = np.sort(err); y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=label, color=color, linewidth=2.4)
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.6, label='AMCL 0.3 m')
    ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    ax.set_xlabel('Localization error (m)'); ax.set_ylabel('Cumulative fraction')
    ax.set_title('Split A (random): synthetic data shifts the\nwhole CDF left  '
                 '(−19% median)')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

    # Panel 2 — median bars across splits
    ax = axes[1]
    groups = ['Split A\n(random)', 'Split C\n(morning→evening)']
    real_med = [np.median(A_real), np.median(C_real)]
    syn_med  = [np.median(A_syn),  np.median(C_syn)]
    xpos = np.arange(len(groups)); bw = 0.35
    b1 = ax.bar(xpos - bw/2, real_med, bw, label='real only', color='#888888')
    b2 = ax.bar(xpos + bw/2, syn_med, bw, label='real + synth', color='#2ca02c')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f'{b.get_height():.3f}', ha='center', fontsize=9)
    # delta annotations
    for i in range(len(groups)):
        d = (syn_med[i] - real_med[i]) / real_med[i] * 100
        col = '#2ca02c' if d < 0 else '#c0392b'
        ax.text(xpos[i], max(real_med[i], syn_med[i]) + 0.18,
                f'{d:+.0f}%', ha='center', fontsize=12, fontweight='bold',
                color=col)
    ax.set_xticks(xpos); ax.set_xticklabels(groups)
    ax.set_ylabel('Median error (m)'); ax.set_ylim(0, 2.3)
    ax.set_title('Helps on random split, slightly hurts on transfer\n'
                 '(GP fit on morning → carries morning bias)')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('GP-kriging synthetic data: fills the spatial coverage gap '
                 '(1449 real positions → +5000 free-space samples)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'synth_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ synth_ablation.png')


def main():
    print(f'[fig] writing to {FIG_DIR}')
    fig_error_cdf()
    fig_error_boxplot()
    fig_cascade_scatter()
    fig_cascade_random100()
    fig_cascade_worst10()
    fig_ladder_bar()
    fig_data_coverage()
    fig_region_reliability()
    fig_error_vs_aps_and_drift()
    fig_synth_ablation()
    print('Done.')


if __name__ == '__main__':
    main()
