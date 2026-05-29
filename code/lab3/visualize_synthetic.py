"""Sanity-check the synthetic data:
  1. For top-4 BSSIDs, plot side-by-side: real RSSI scatter vs GP predicted heatmap
  2. Real vs synthetic spatial coverage on the map (synthetic should fill gaps)
  3. APs-per-scan histogram comparison
  4. Per-BSSID detection-rate comparison (real vs synthetic)
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                     'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import data
import synthetic

ROOT = Path(__file__).resolve().parents[2]
MAP_YAML = ROOT / 'map' / 'psquare.yaml'
OUT_DIR = Path(__file__).parent / 'outputs'
PLOTS_DIR = OUT_DIR / 'plots_synthetic'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_map():
    with open(MAP_YAML) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(MAP_YAML.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution'], pgm.shape[1], pgm.shape[0]


def main():
    print('[load] real records + synthetic data...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)
    morning = [r for r in records if r['session'] == 'morning']
    X_real, y_real, sess = data.build_arrays(morning, bssids)

    synth_npz = OUT_DIR / 'synthetic_morning_5000.npz'
    sd = np.load(synth_npz, allow_pickle=True)
    X_synth, y_synth = sd['X_synth'], sd['y_synth']
    print(f'       real: {len(X_real)},  synth: {len(X_synth)}')

    pgm, ox, oy, res, W, H = load_map()

    # ── 1. Spatial coverage comparison ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pts, title, color in [
        (axes[0], y_real, f'真實 morning ({len(y_real)} positions)', 'forestgreen'),
        (axes[1], y_synth, f'合成 ({len(y_synth)} positions)', 'crimson'),
    ]:
        ax.imshow(pgm, cmap='gray', extent=[ox, ox + W * res, oy, oy + H * res])
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c=color, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    fig.suptitle('Spatial coverage: real vs synthetic\n'
                 '合成資料填滿 free space,真實只在 trajectory 上')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '01_spatial_coverage.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote 01_spatial_coverage.png')

    # ── 2. APs-per-scan histogram ────────────────────────────────
    real_aps_per = (X_real > -99.5).sum(axis=1)
    synth_aps_per = (X_synth > -99.5).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_aps_per, bins=range(0, 60), alpha=0.6, color='forestgreen',
             label=f'真實 (mean {real_aps_per.mean():.1f})')
    ax.hist(synth_aps_per, bins=range(0, 60), alpha=0.6, color='crimson',
             label=f'合成 (mean {synth_aps_per.mean():.1f})')
    ax.set_xlabel('AP / scan')
    ax.set_ylabel('count')
    ax.set_title('Q: 合成 scan 有多少 AP?(理想要跟真實接近)')
    ax.legend()
    plt.savefig(PLOTS_DIR / '02_aps_per_scan.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote 02_aps_per_scan.png')

    # ── 3. Per-BSSID detection rate comparison ──────────────────
    real_det_rate = (X_real > -99.5).mean(axis=0)         # 80 維
    synth_det_rate = (X_synth > -99.5).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(real_det_rate, synth_det_rate, s=15, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', label='ideal')
    ax.set_xlabel('real morning detection rate')
    ax.set_ylabel('synthetic detection rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Per-BSSID detection rate: 合成 vs 真實\n'
                 '理想:點落在對角線上;偏上 = 合成過度樂觀')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / '03_detection_rate.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote 03_detection_rate.png')

    # ── 4. Per-BSSID RSSI heatmaps (top 4) ───────────────────────
    # Find top-4 BSSIDs by detection count in real data
    detection_counts = (X_real > -99.5).sum(axis=0)
    top_bssid_idx = np.argsort(-detection_counts)[:4]
    # Load SSID labels directly from jsonl (data.py doesn't keep them)
    import glob, json
    bssid_ssid_map = {}
    for jf in glob.glob(str(ROOT / 'wifi' / 'wifi_20260517_*.jsonl')):
        with open(jf, encoding='utf-8') as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                for ap in d.get('aps', []):
                    b = (ap.get('bssid') or '').upper()
                    s = ap.get('ssid', '')
                    if b and s and b not in bssid_ssid_map:
                        bssid_ssid_map[b] = s

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    for row, j in enumerate(top_bssid_idx):
        bssid = bssids[j]
        ssid = bssid_ssid_map.get(bssid, bssid[:8])[:20]
        # Real
        ax = axes[row, 0]
        ax.imshow(pgm, cmap='gray', extent=[ox, ox + W * res, oy, oy + H * res])
        seen_real = X_real[:, j] > -99.5
        sc = ax.scatter(y_real[seen_real, 0], y_real[seen_real, 1],
                          c=X_real[seen_real, j], cmap='RdYlGn', vmin=-90, vmax=-40,
                          s=8, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='RSSI (dBm)')
        ax.set_title(f'真實: {ssid}\n({seen_real.sum()} detections)')
        ax.set_aspect('equal')
        # Synthetic
        ax = axes[row, 1]
        ax.imshow(pgm, cmap='gray', extent=[ox, ox + W * res, oy, oy + H * res])
        seen_synth = X_synth[:, j] > -99.5
        sc = ax.scatter(y_synth[seen_synth, 0], y_synth[seen_synth, 1],
                          c=X_synth[seen_synth, j], cmap='RdYlGn', vmin=-90, vmax=-40,
                          s=4, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='RSSI (dBm)')
        ax.set_title(f'合成: {ssid}\n({seen_synth.sum()} detections)')
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '04_per_ap_heatmap.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote 04_per_ap_heatmap.png')

    # ── 5. Mean RSSI comparison per BSSID ────────────────────────
    real_means = []
    synth_means = []
    for j in range(len(bssids)):
        sr = X_real[:, j]; mr = sr[sr > -99.5]
        ss = X_synth[:, j]; ms = ss[ss > -99.5]
        if len(mr) >= 5 and len(ms) >= 5:
            real_means.append(mr.mean())
            synth_means.append(ms.mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(real_means, synth_means, s=15, alpha=0.6)
    ax.plot([-95, -30], [-95, -30], 'k--', label='ideal')
    ax.set_xlabel('real mean RSSI (dBm)')
    ax.set_ylabel('synthetic mean RSSI (dBm)')
    ax.set_title('Per-BSSID mean RSSI: 合成 vs 真實')
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / '05_mean_rssi.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote 05_mean_rssi.png')

    print(f'\n✓ all plots in {PLOTS_DIR}')


if __name__ == '__main__':
    main()
