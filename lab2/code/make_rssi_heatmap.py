#!/usr/bin/env python3
"""make_rssi_heatmap — full RSSI analysis suite from wifi jsonl.

Generates multiple visualizations + statistics for lab submission:

  heatmap_TOP/                       per-AP heatmap, top-N strongest by samples
    heatmap_NN_<ssid>.png            single-AP RSSI gradient overlaid on map
  heatmap_combined_best.png          for each cell, the BEST RSSI from any top-N AP
  heatmap_sample_density.png         how many WiFi scans hit each cell (coverage map)
  heatmap_dominant_ap.png            for each cell, which AP has strongest signal
                                     (color-coded by AP)
  heatmap_stats.csv                  per-AP statistics (peak rssi, samples, etc.)

Uses chunked Gaussian interpolation to fit in memory for 400+ AP × 437x489 map.

Usage:
  python3 ~/ros2_ws/lab2_configs/make_rssi_heatmap.py
  python3 ~/ros2_ws/lab2_configs/make_rssi_heatmap.py --top-n 8 --bandwidth 1.0
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


def chunked_gaussian_interp(samples_xy, samples_v, grid_xy, bandwidth, chunk=4000):
    """Memory-friendly gaussian kernel weighted average."""
    H, W, _ = grid_xy.shape
    flat = grid_xy.reshape(-1, 2)
    N = len(flat)
    out = np.empty(N, dtype=float)
    inv_2bw2 = 1.0 / (2 * bandwidth * bandwidth)
    for i in range(0, N, chunk):
        block = flat[i:i+chunk]                       # (b, 2)
        diff = block[:, None, :] - samples_xy[None, :, :]   # (b, M, 2)
        d2 = (diff ** 2).sum(axis=-1)                       # (b, M)
        w = np.exp(-d2 * inv_2bw2)                          # (b, M)
        w_sum = w.sum(axis=1)
        wv = (w * samples_v[None, :]).sum(axis=1)
        out[i:i+chunk] = np.where(w_sum > 1e-6, wv / w_sum, np.nan)
    return out.reshape(H, W)


def nearest_sample_distance(samples_xy, grid_xy, chunk=4000):
    """For each grid cell, distance (m) to nearest sample."""
    H, W, _ = grid_xy.shape
    flat = grid_xy.reshape(-1, 2)
    N = len(flat)
    out = np.empty(N, dtype=float)
    for i in range(0, N, chunk):
        block = flat[i:i+chunk]
        diff = block[:, None, :] - samples_xy[None, :, :]
        d2 = (diff ** 2).sum(axis=-1)
        out[i:i+chunk] = np.sqrt(d2.min(axis=1))
    return out.reshape(H, W)


def rssi_to_rgba(grid, vmin=-90, vmax=-30, alpha=180):
    H, W = grid.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    valid = ~np.isnan(grid)
    norm = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    # red(0) → yellow(0.5) → green(1)
    r = np.where(norm > 0.5, (1 - norm) * 2, 1.0)
    g = np.where(norm > 0.5, 1.0, norm * 2)
    rgba[..., 0] = np.where(valid, (r * 255).astype(np.uint8), 0)
    rgba[..., 1] = np.where(valid, (g * 255).astype(np.uint8), 0)
    rgba[..., 3] = np.where(valid, alpha, 0)
    return rgba


def density_to_rgba(grid, alpha=180):
    H, W = grid.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if grid.max() == 0:
        return rgba
    norm = np.log1p(grid) / np.log1p(grid.max())  # log scale
    # purple → blue → green → yellow
    valid = grid > 0
    rgba[..., 0] = np.where(valid, (norm * 255).astype(np.uint8), 0)
    rgba[..., 1] = np.where(valid, ((1 - abs(norm - 0.5) * 2) * 255).astype(np.uint8), 0)
    rgba[..., 2] = np.where(valid, ((1 - norm) * 255).astype(np.uint8), 0)
    rgba[..., 3] = np.where(valid, alpha, 0)
    return rgba


def make_legend_text(img: Image.Image, lines: list, x=10, y=10):
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((x, y + i * 14), line, fill='black')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wifi-dir', default=os.path.expanduser('~/ros2_ws/wifi_data'))
    ap.add_argument('--map-yaml', default=os.path.expanduser('~/ros2_ws/maps/psquare.yaml'))
    ap.add_argument('--output-dir', default=os.path.expanduser('~/ros2_ws/wifi_data'))
    ap.add_argument('--top-n', type=int, default=8)
    ap.add_argument('--bandwidth', type=float, default=1.0)
    ap.add_argument('--max-sample-dist', type=float, default=1.5,
                    help='Hide heatmap cells whose nearest sample > N meters away')
    ap.add_argument('--unknown-pixel', type=int, default=128,
                    help='Map pixel value treated as unknown / unexplored')
    args = ap.parse_args()

    # ── Map ──────────────────────────────────────────
    with open(args.map_yaml) as f:
        meta = yaml.safe_load(f)
    res = float(meta['resolution'])
    ox, oy = meta['origin'][0], meta['origin'][1]
    map_dir = os.path.dirname(args.map_yaml)
    pgm = np.array(Image.open(os.path.join(map_dir, meta['image'])))
    H, W = pgm.shape
    xs = ox + (np.arange(W) + 0.5) * res
    ys = oy + (np.arange(H) + 0.5) * res
    ys = ys[::-1]
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.stack([gx, gy], axis=-1)

    # ── Load wifi ────────────────────────────────────
    ap_samples = defaultdict(list)   # bssid -> [(x,y,rssi,ssid)]
    n_records = 0
    for jf in sorted(glob.glob(os.path.join(args.wifi_dir, 'wifi_*.jsonl'))):
        if os.path.getsize(jf) == 0:
            continue
        with open(jf) as f:
            for line in f:
                try: r = json.loads(line)
                except: continue
                p = r.get('pose', {})
                if p.get('frame_id') != 'map': continue
                x, y = p.get('x'), p.get('y')
                if x is None or y is None: continue
                for apx in r.get('aps', []):
                    ap_samples[apx['bssid']].append(
                        (x, y, apx['rssi'], apx.get('ssid', '')))
                n_records += 1
    print(f'loaded {n_records} records, {len(ap_samples)} unique APs')

    # ── Sort APs by sample count ────────────────────
    sorted_aps = sorted(ap_samples.items(),
                        key=lambda kv: len(kv[1]), reverse=True)
    top = sorted_aps[:args.top_n]

    rgb_base = np.stack([pgm]*3, axis=-1).astype(np.uint8)
    base_img = Image.fromarray(rgb_base).convert('RGBA')

    # Geographic mask: only show heatmap where map is observed (not unknown).
    # Cartographer marks unobserved cells with args.unknown_pixel (typically 128).
    geo_mask = pgm != args.unknown_pixel
    print(f'geo mask: {geo_mask.sum()}/{geo_mask.size} cells observed '
          f'({100*geo_mask.mean():.1f}% of map)')

    # ── Per-AP heatmaps ──────────────────────────────
    out_top_dir = os.path.join(args.output_dir, 'heatmap_TOP')
    os.makedirs(out_top_dir, exist_ok=True)
    stats_rows = []
    print(f'\n=== Top {len(top)} APs ===')
    for i, (bssid, samples) in enumerate(top, 1):
        ssid = samples[0][3] or '(empty)'
        xy = np.array([(s[0], s[1]) for s in samples], dtype=float)
        rs = np.array([s[2] for s in samples], dtype=float)
        peak_idx = np.argmax(rs)
        stats_rows.append({
            'rank': i, 'ssid': ssid, 'bssid': bssid,
            'samples': len(samples),
            'rssi_min': int(rs.min()), 'rssi_max': int(rs.max()),
            'rssi_mean': round(rs.mean(), 1), 'rssi_std': round(rs.std(), 1),
            'peak_x': round(xy[peak_idx, 0], 2),
            'peak_y': round(xy[peak_idx, 1], 2),
        })
        print(f'  {i:2d}. {ssid:20s}  n={len(samples):4d}  '
              f'rssi [{int(rs.min()):3d},{int(rs.max()):3d}]  '
              f'avg {rs.mean():.1f}  std {rs.std():.1f}')

        heat = chunked_gaussian_interp(xy, rs, grid_xy, args.bandwidth)
        # Mask: hide cells too far from any sample, or outside observed map
        dist = nearest_sample_distance(xy, grid_xy)
        mask = geo_mask & (dist <= args.max_sample_dist)
        heat = np.where(mask, heat, np.nan)
        rgba = rssi_to_rgba(heat)
        overlay = Image.fromarray(rgba, mode='RGBA')
        result = Image.alpha_composite(base_img, overlay)
        # annotate
        make_legend_text(result, [
            f'#{i}  {ssid}',
            f'bssid: {bssid}',
            f'samples: {len(samples)}',
            f'rssi: {int(rs.min())} .. {int(rs.max())} dBm  (avg {rs.mean():.1f})',
            f'peak at ({xy[peak_idx,0]:.2f}, {xy[peak_idx,1]:.2f})',
            '',
            'green=strong  yellow=med  red=weak',
        ])
        safe = ''.join(c if c.isalnum() else '_' for c in ssid)[:20]
        out = os.path.join(out_top_dir, f'heatmap_{i:02d}_{safe}.png')
        result.convert('RGB').save(out)

    # ── Sample density map ──────────────────────────
    print('\nbuilding sample density map...')
    density = np.zeros((H, W), dtype=int)
    for samples in ap_samples.values():
        # only count once per unique (x,y) (per record) — but we don't have
        # per-record bssid grouping; using all AP samples = scans × aps,
        # which still reflects coverage. Below counts hits per cell.
        for x, y, _, _ in samples:
            cx = int((x - ox) / res)
            cy = H - 1 - int((y - oy) / res)
            if 0 <= cx < W and 0 <= cy < H:
                density[cy, cx] += 1
    rgba = density_to_rgba(density)
    overlay = Image.fromarray(rgba, mode='RGBA')
    result = Image.alpha_composite(base_img, overlay)
    make_legend_text(result, [
        f'Sample density (log scale)',
        f'max {density.max()} hits in single cell',
        f'covered cells: {(density > 0).sum()}',
        'blue=sparse  green=mid  yellow=dense',
    ])
    out = os.path.join(args.output_dir, 'heatmap_sample_density.png')
    result.convert('RGB').save(out)
    print(f'  → {out}')

    # ── Combined "best AP" RSSI (max over top-N) ────
    print('\nbuilding combined-best heatmap...')
    # Union of all top-N samples for shared mask
    all_top_xy = np.concatenate([
        np.array([(s[0], s[1]) for s in samples])
        for _, samples in top
    ], axis=0)
    union_dist = nearest_sample_distance(all_top_xy, grid_xy)
    union_mask = geo_mask & (union_dist <= args.max_sample_dist)

    best = np.full((H, W), -np.inf)
    for bssid, samples in top:
        xy = np.array([(s[0], s[1]) for s in samples], dtype=float)
        rs = np.array([s[2] for s in samples], dtype=float)
        heat = chunked_gaussian_interp(xy, rs, grid_xy, args.bandwidth)
        # per-cell max
        valid = ~np.isnan(heat)
        best = np.where(valid & (heat > best), heat, best)
    best = np.where(np.isinf(best), np.nan, best)
    best = np.where(union_mask, best, np.nan)
    rgba = rssi_to_rgba(best)
    overlay = Image.fromarray(rgba, mode='RGBA')
    result = Image.alpha_composite(base_img, overlay)
    make_legend_text(result, [
        'Best signal (max over top-N APs)',
        f'Top {len(top)} APs considered',
        'green=strong best signal  red=weak best',
    ])
    out = os.path.join(args.output_dir, 'heatmap_combined_best.png')
    result.convert('RGB').save(out)
    print(f'  → {out}')

    # ── Dominant AP map (which AP wins per cell) ────
    print('\nbuilding dominant-AP map...')
    import colorsys
    dom = np.full((H, W), -1, dtype=int)
    best_rssi = np.full((H, W), -np.inf)
    for ap_idx, (bssid, samples) in enumerate(top):
        xy = np.array([(s[0], s[1]) for s in samples], dtype=float)
        rs = np.array([s[2] for s in samples], dtype=float)
        heat = chunked_gaussian_interp(xy, rs, grid_xy, args.bandwidth)
        valid = ~np.isnan(heat)
        winning = valid & (heat > best_rssi)
        dom = np.where(winning, ap_idx, dom)
        best_rssi = np.where(winning, heat, best_rssi)
    # Apply union mask (same as combined-best — only show where any top AP near)
    dom = np.where(union_mask, dom, -1)

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for ap_idx in range(len(top)):
        h = ap_idx / max(1, len(top))
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        mask = dom == ap_idx
        rgba[..., 0] = np.where(mask, int(r * 255), rgba[..., 0])
        rgba[..., 1] = np.where(mask, int(g * 255), rgba[..., 1])
        rgba[..., 2] = np.where(mask, int(b * 255), rgba[..., 2])
        rgba[..., 3] = np.where(mask, 160, rgba[..., 3])
    overlay = Image.fromarray(rgba, mode='RGBA')
    result = Image.alpha_composite(base_img, overlay)
    legend = ['Dominant AP per cell:']
    for ap_idx, (bssid, samples) in enumerate(top):
        legend.append(f'  #{ap_idx+1} {samples[0][3]}')
    make_legend_text(result, legend)
    out = os.path.join(args.output_dir, 'heatmap_dominant_ap.png')
    result.convert('RGB').save(out)
    print(f'  → {out}')

    # ── Stats CSV ────────────────────────────────────
    out_csv = os.path.join(args.output_dir, 'heatmap_stats.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        for row in stats_rows:
            w.writerow(row)
    print(f'\n✓ stats csv → {out_csv}')
    print(f'\ndone. all outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
