"""Presentation figure for slide 18 (how GP-synth fills the coverage hole).
Reads the cached synthetic set (no slow GP re-fit) and renders, in English:
  left  = real morning scans (path-shaped, upper region empty)
  mid   = + GP-synthetic scans (free space filled)
  right = one AP's RSSI — sparse real over the smooth GP-filled field
Saves outputs/figures/synth_coverage.png so the deck's figure dir stays self-contained.
"""
import sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import data

ROOT = Path(__file__).resolve().parents[1]
MAP_YAML = ROOT / 'map' / 'psquare.yaml'
OUT = Path(__file__).parent / 'outputs'
FIG = OUT / 'figures'


def load_map():
    with open(MAP_YAML) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(MAP_YAML.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution'], pgm.shape[1], pgm.shape[0]


def main():
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    morning = [r for r in records if r['session'] == 'morning']
    X_real, y_real, _ = data.build_arrays(morning, bssids)
    sd = np.load(OUT / 'synthetic_morning_5000.npz', allow_pickle=True)
    X_syn, y_syn = sd['X_synth'], sd['y_synth']
    pgm, ox, oy, res, W, H = load_map()
    ext = [ox, ox + W * res, oy, oy + H * res]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    # ── 1. real coverage ──
    ax[0].imshow(pgm, cmap='gray', extent=ext)
    ax[0].scatter(y_real[:, 0], y_real[:, 1], s=4, c='#1a7a3a', alpha=0.5)
    ax[0].set_title(f'Real scans only ({len(y_real)})\npath-shaped — upper region empty',
                    fontsize=11, fontweight='bold')

    # ── 2. + synthetic coverage ──
    ax[1].imshow(pgm, cmap='gray', extent=ext)
    ax[1].scatter(y_real[:, 0], y_real[:, 1], s=4, c='#1a7a3a', alpha=0.5, label='real')
    ax[1].scatter(y_syn[:, 0], y_syn[:, 1], s=2, c='#d62728', alpha=0.25, label='GP-synth')
    ax[1].set_title(f'+ {len(y_syn)} GP-synthetic scans\nfree space filled',
                    fontsize=11, fontweight='bold')
    ax[1].legend(loc='upper right', fontsize=8, markerscale=2)

    # ── 3. one AP: sparse real over GP-filled RSSI field ──
    j = int(np.argmax((X_real > -99.5).sum(0)))   # most-detected AP
    ax[2].imshow(pgm, cmap='gray', extent=ext)
    sy = X_syn[:, j] > -99.5
    ax[2].scatter(y_syn[sy, 0], y_syn[sy, 1], c=X_syn[sy, j], cmap='RdYlGn',
                  vmin=-90, vmax=-40, s=6, alpha=0.45)
    sr = X_real[:, j] > -99.5
    sc = ax[2].scatter(y_real[sr, 0], y_real[sr, 1], c=X_real[sr, j], cmap='RdYlGn',
                       vmin=-90, vmax=-40, s=22, edgecolor='black', linewidth=0.3)
    plt.colorbar(sc, ax=ax[2], label='RSSI (dBm)', fraction=0.046)
    ax[2].set_title('One AP: real dots (outlined) over\nthe GP-filled smooth RSSI field',
                    fontsize=11, fontweight='bold')

    for a in ax:
        a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_aspect('equal')

    fig.suptitle('GP-kriging fills the coverage hole with physically grounded scans '
                 '(fit on TRAIN positions only)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / 'synth_coverage.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
