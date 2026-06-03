#!/usr/bin/env python3
"""make_morning_evening_diff — RSSI 早晚差異分析.

Compares morning (2026-05-17) vs evening (2026-05-23) wifi data:
  - 同一個 AP 在同位置 RSSI 變了多少
  - 哪些 AP 早晚穩定,哪些 fluctuating(常見 SSID = neighbour wifi 可能被人開關)
  - Diff heatmap: evening_rssi - morning_rssi (dBm)

Outputs (~/ros2_ws/wifi_data/diff_morning_evening/):
  diff_TOP/diff_NN_<ssid>.png       per-AP diff heatmap
  diff_stats.csv                    morning/evening avg + delta per AP
  diff_summary.png                  side-by-side: morning | evening | diff
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
    H, W, _ = grid_xy.shape
    flat = grid_xy.reshape(-1, 2)
    N = len(flat)
    out = np.empty(N, dtype=float)
    inv_2bw2 = 1.0 / (2 * bandwidth * bandwidth)
    for i in range(0, N, chunk):
        block = flat[i:i+chunk]
        diff = block[:, None, :] - samples_xy[None, :, :]
        d2 = (diff ** 2).sum(axis=-1)
        w = np.exp(-d2 * inv_2bw2)
        w_sum = w.sum(axis=1)
        wv = (w * samples_v[None, :]).sum(axis=1)
        out[i:i+chunk] = np.where(w_sum > 1e-6, wv / w_sum, np.nan)
    return out.reshape(H, W)


def nearest_sample_distance(samples_xy, grid_xy, chunk=4000):
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


def diff_to_rgba(diff, vmin=-15, vmax=15, alpha=200):
    """blue = signal got weaker (evening worse), red = stronger (evening better)."""
    H, W = diff.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    valid = ~np.isnan(diff)
    norm = np.clip((diff - vmin) / (vmax - vmin), 0, 1)  # 0..1
    # diverging colormap: blue(0) -> white(0.5) -> red(1)
    r = np.where(norm > 0.5, 255, (norm*2 * 255).astype(int))
    b = np.where(norm < 0.5, 255, ((1-norm)*2 * 255).astype(int))
    g = np.where(norm > 0.5, ((1-norm)*2 * 255).astype(int),
                 (norm*2 * 255).astype(int))
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = np.where(valid, alpha, 0)
    return rgba


def load_jsonl_files(paths):
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pose = d.get('pose', {})
                if pose.get('frame_id') != 'map':
                    continue
                x, y = pose.get('x'), pose.get('y')
                if x is None or y is None:
                    continue
                for ap in d.get('aps', []):
                    bssid = ap.get('bssid')
                    rssi = ap.get('rssi')
                    if not bssid or rssi is None:
                        continue
                    records.append({
                        'bssid': bssid.upper(),
                        'ssid': ap.get('ssid', ''),
                        'rssi': float(rssi),
                        'x': float(x), 'y': float(y),
                    })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wifi-dir', default=os.path.expanduser('~/ros2_ws/wifi_data'))
    ap.add_argument('--map-yaml', default=os.path.expanduser('~/ros2_ws/maps/psquare.yaml'))
    ap.add_argument('--out-dir', default=os.path.expanduser('~/ros2_ws/wifi_data/diff_morning_evening'))
    ap.add_argument('--morning-glob', default='wifi_20260517_*.jsonl')
    ap.add_argument('--evening-glob', default='wifi_20260523_*.jsonl')
    ap.add_argument('--top-n', type=int, default=8)
    ap.add_argument('--bandwidth', type=float, default=1.0)
    ap.add_argument('--min-samples', type=int, default=30,
                    help='AP must have >= this many samples in BOTH morning and evening')
    ap.add_argument('--max-sample-dist', type=float, default=1.5,
                    help='Hide diff cells whose nearest sample > N meters away (both batches)')
    ap.add_argument('--unknown-pixel', type=int, default=128,
                    help='Map pixel value treated as unknown / unexplored')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'diff_TOP'), exist_ok=True)

    # load map
    with open(args.map_yaml) as f:
        meta = yaml.safe_load(f)
    pgm_path = os.path.join(os.path.dirname(args.map_yaml), meta['image'])
    map_img = Image.open(pgm_path).convert('RGB')
    pgm_gray = np.array(Image.open(pgm_path))
    W, H = map_img.size
    res = meta['resolution']
    ox, oy = meta['origin'][0], meta['origin'][1]
    geo_mask = pgm_gray != args.unknown_pixel
    # grid xy in map frame (each pixel center)
    xs = ox + (np.arange(W) + 0.5) * res
    ys = oy + (H - np.arange(H) - 0.5) * res    # image y inverted
    grid_xy = np.stack(np.meshgrid(xs, ys), axis=-1)   # (H, W, 2)

    morning_files = sorted(glob.glob(os.path.join(args.wifi_dir, args.morning_glob)))
    evening_files = sorted(glob.glob(os.path.join(args.wifi_dir, args.evening_glob)))
    print(f'morning: {len(morning_files)} files, evening: {len(evening_files)} files')

    morning = load_jsonl_files(morning_files)
    evening = load_jsonl_files(evening_files)
    print(f'morning: {len(morning)} AP-records, evening: {len(evening)} AP-records')

    # group by bssid
    by_bssid_m = defaultdict(list)
    by_bssid_e = defaultdict(list)
    for r in morning:
        by_bssid_m[r['bssid']].append(r)
    for r in evening:
        by_bssid_e[r['bssid']].append(r)

    # pick APs that appear sufficiently in BOTH
    common = []
    for bssid in set(by_bssid_m) & set(by_bssid_e):
        if len(by_bssid_m[bssid]) >= args.min_samples and len(by_bssid_e[bssid]) >= args.min_samples:
            common.append(bssid)
    # sort by total samples descending
    common.sort(key=lambda b: len(by_bssid_m[b]) + len(by_bssid_e[b]), reverse=True)
    top = common[:args.top_n]
    print(f'common APs (>={args.min_samples} samples each): {len(common)}, plotting top {len(top)}')

    # stats CSV
    stats = []
    for bssid in common:
        rm = by_bssid_m[bssid]
        re = by_bssid_e[bssid]
        m_avg = np.mean([r['rssi'] for r in rm])
        e_avg = np.mean([r['rssi'] for r in re])
        m_std = np.std([r['rssi'] for r in rm])
        e_std = np.std([r['rssi'] for r in re])
        stats.append({
            'ssid': rm[0]['ssid'],
            'bssid': bssid,
            'morning_samples': len(rm),
            'evening_samples': len(re),
            'morning_avg_rssi': round(m_avg, 2),
            'evening_avg_rssi': round(e_avg, 2),
            'morning_std_rssi': round(m_std, 2),
            'evening_std_rssi': round(e_std, 2),
            'delta_avg': round(e_avg - m_avg, 2),
        })
    stats.sort(key=lambda s: abs(s['delta_avg']), reverse=True)
    stats_csv = os.path.join(args.out_dir, 'diff_stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print(f'wrote {stats_csv}  ({len(stats)} APs)')

    # Top-changed (biggest delta) summary
    print('\nTop 10 biggest morning→evening RSSI shifts:')
    for s in stats[:10]:
        sign = '+' if s['delta_avg'] >= 0 else ''
        print(f'  {sign}{s["delta_avg"]:>6} dBm  {s["ssid"]:<24} ({s["bssid"]})')

    # diff heatmaps for top-N
    for rank, bssid in enumerate(top):
        rm = by_bssid_m[bssid]
        re = by_bssid_e[bssid]
        ssid = rm[0]['ssid']
        safe = ''.join(c if c.isalnum() else '_' for c in ssid)[:20] or 'unk'

        m_xy = np.array([[r['x'], r['y']] for r in rm])
        m_v = np.array([r['rssi'] for r in rm])
        e_xy = np.array([[r['x'], r['y']] for r in re])
        e_v = np.array([r['rssi'] for r in re])

        m_grid = chunked_gaussian_interp(m_xy, m_v, grid_xy, args.bandwidth)
        e_grid = chunked_gaussian_interp(e_xy, e_v, grid_xy, args.bandwidth)
        diff_grid = e_grid - m_grid
        # Mask: need both morning and evening samples close to the cell + in mapped area
        m_dist = nearest_sample_distance(m_xy, grid_xy)
        e_dist = nearest_sample_distance(e_xy, grid_xy)
        valid = geo_mask & (m_dist <= args.max_sample_dist) & (e_dist <= args.max_sample_dist)
        diff_grid = np.where(valid, diff_grid, np.nan)

        rgba = diff_to_rgba(diff_grid, vmin=-15, vmax=15)
        overlay = Image.fromarray(rgba, 'RGBA')
        base = map_img.copy().convert('RGBA')
        out = Image.alpha_composite(base, overlay)
        draw = ImageDraw.Draw(out)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        except OSError:
            font = ImageFont.load_default()
        delta = np.nanmean(diff_grid)
        title = f'{ssid} ({bssid})  evening - morning  avg={delta:+.1f} dBm'
        draw.rectangle([(2, 2), (W-2, 22)], fill=(255, 255, 255, 200))
        draw.text((6, 4), title, fill=(0, 0, 0), font=font)
        # legend
        draw.rectangle([(2, H-22), (W-2, H-2)], fill=(255, 255, 255, 200))
        draw.text((6, H-20),
                  'blue = evening WEAKER  |  red = evening STRONGER  |  +/- 15 dBm',
                  fill=(0, 0, 0), font=font)

        out_path = os.path.join(args.out_dir, 'diff_TOP', f'diff_{rank:02d}_{safe}.png')
        out.convert('RGB').save(out_path)
        print(f'  [{rank+1}/{len(top)}] {ssid:<22} delta={delta:+.2f} dBm  →  {out_path}')

    print('\n✓ done')


if __name__ == '__main__':
    main()
