"""整體早晚對比(非 per-AP)— 把所有 top-N AP 在每格的『最強訊號』疊起來比較.

每格:
  morning_best = max over top-N APs (morning 樣本內插)
  evening_best = max over top-N APs (evening 樣本內插)
  diff = evening_best - morning_best

輸出 2 張:
  overall_diff_best.png     evening best - morning best  (整體訊號變強或變弱)
  overall_diff_mean.png     對全部 top-N AP 的 diff 取平均(訊號平均強度變化)
"""
import argparse
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


def diff_to_rgba(diff, vmin=-6, vmax=6, alpha=200):
    H, W = diff.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    valid = ~np.isnan(diff)
    norm = np.clip((diff - vmin) / (vmax - vmin), 0, 1)
    r = np.where(norm > 0.5, 255, (norm * 2 * 255).astype(int))
    b = np.where(norm < 0.5, 255, ((1 - norm) * 2 * 255).astype(int))
    g = np.where(norm > 0.5,
                 ((1 - norm) * 2 * 255).astype(int),
                 (norm * 2 * 255).astype(int))
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = np.where(valid, alpha, 0)
    return rgba


def load_jsonl(paths):
    by_ap_session = defaultdict(lambda: defaultdict(list))   # bssid -> session -> [(x,y,rssi,ssid)]
    for p in paths:
        session = 'morning' if '20260517' in p else 'evening'
        for line in open(p):
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
                bssid = ap.get('bssid', '').upper()
                rssi = ap.get('rssi')
                if not bssid or rssi is None:
                    continue
                by_ap_session[bssid][session].append((float(x), float(y), float(rssi), ap.get('ssid', '')))
    return by_ap_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wifi-dir', default=os.path.expanduser('~/ros2_ws/wifi_data'))
    ap.add_argument('--map-yaml', default=os.path.expanduser('~/ros2_ws/maps/psquare.yaml'))
    ap.add_argument('--out-dir', default=os.path.expanduser('~/ros2_ws/wifi_data/overall_diff'))
    ap.add_argument('--top-n', type=int, default=8)
    ap.add_argument('--bandwidth', type=float, default=1.0)
    ap.add_argument('--min-samples', type=int, default=30)
    ap.add_argument('--max-sample-dist', type=float, default=1.5)
    ap.add_argument('--unknown-pixel', type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load map
    with open(args.map_yaml) as f:
        meta = yaml.safe_load(f)
    pgm_path = os.path.join(os.path.dirname(args.map_yaml), meta['image'])
    pgm = np.array(Image.open(pgm_path))
    H, W = pgm.shape
    res = float(meta['resolution'])
    ox, oy = meta['origin'][0], meta['origin'][1]
    xs = ox + (np.arange(W) + 0.5) * res
    ys = oy + (np.arange(H) + 0.5) * res
    ys = ys[::-1]
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.stack([gx, gy], axis=-1)
    geo_mask = pgm != args.unknown_pixel

    rgb_base = np.stack([pgm]*3, axis=-1).astype(np.uint8)
    base_img = Image.fromarray(rgb_base).convert('RGBA')

    # Load all wifi
    files = sorted(glob.glob(os.path.join(args.wifi_dir, 'wifi_2026*.jsonl')))
    files = [f for f in files if os.path.getsize(f) > 0]
    by_ap = load_jsonl(files)

    # Find APs with enough samples in BOTH sessions
    qualified = []
    for bssid, sessions in by_ap.items():
        if (len(sessions['morning']) >= args.min_samples and
            len(sessions['evening']) >= args.min_samples):
            qualified.append((bssid, sessions))
    qualified.sort(key=lambda kv: len(kv[1]['morning']) + len(kv[1]['evening']),
                   reverse=True)
    top = qualified[:args.top_n]
    print(f'qualified APs (>={args.min_samples} in both): {len(qualified)}, using top {len(top)}')

    # Compute morning + evening heatmaps for each AP
    m_grids = []
    e_grids = []
    all_m_xy = []
    all_e_xy = []
    ap_labels = []
    for bssid, sessions in top:
        ssid = sessions['morning'][0][3]
        ap_labels.append(ssid)

        m_xy = np.array([(s[0], s[1]) for s in sessions['morning']])
        m_v = np.array([s[2] for s in sessions['morning']])
        e_xy = np.array([(s[0], s[1]) for s in sessions['evening']])
        e_v = np.array([s[2] for s in sessions['evening']])

        m_grids.append(chunked_gaussian_interp(m_xy, m_v, grid_xy, args.bandwidth))
        e_grids.append(chunked_gaussian_interp(e_xy, e_v, grid_xy, args.bandwidth))
        all_m_xy.append(m_xy)
        all_e_xy.append(e_xy)
        print(f'  {ssid:<24} morning={len(m_v):4d} evening={len(e_v):4d}')

    all_m_xy = np.concatenate(all_m_xy)
    all_e_xy = np.concatenate(all_e_xy)
    m_dist = nearest_sample_distance(all_m_xy, grid_xy)
    e_dist = nearest_sample_distance(all_e_xy, grid_xy)
    valid_mask = geo_mask & (m_dist <= args.max_sample_dist) & (e_dist <= args.max_sample_dist)

    # ── Best-vs-best diff ──
    m_best = np.full((H, W), -np.inf)
    e_best = np.full((H, W), -np.inf)
    for mg, eg in zip(m_grids, e_grids):
        m_v = ~np.isnan(mg)
        e_v = ~np.isnan(eg)
        m_best = np.where(m_v & (mg > m_best), mg, m_best)
        e_best = np.where(e_v & (eg > e_best), eg, e_best)
    m_best = np.where(np.isinf(m_best), np.nan, m_best)
    e_best = np.where(np.isinf(e_best), np.nan, e_best)
    diff_best = e_best - m_best
    diff_best = np.where(valid_mask, diff_best, np.nan)

    rgba = diff_to_rgba(diff_best, vmin=-6, vmax=6)
    overlay = Image.fromarray(rgba, 'RGBA')
    out = Image.alpha_composite(base_img, overlay)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    avg = np.nanmean(diff_best)
    draw.rectangle([(2, 2), (W-2, 38)], fill=(255, 255, 255, 220))
    draw.text((6, 4), 'Overall BEST-signal change  (evening - morning)', fill=(0, 0, 0), font=font)
    draw.text((6, 20), f'top-{len(top)} APs combined  avg={avg:+.2f} dBm',
              fill=(0, 0, 0), font=font)
    draw.rectangle([(2, H-22), (W-2, H-2)], fill=(255, 255, 255, 220))
    draw.text((6, H-20),
              'blue = best signal got WEAKER  |  red = best signal got STRONGER  |  +-6 dBm',
              fill=(0, 0, 0), font=font)
    out_path = os.path.join(args.out_dir, 'overall_diff_best.png')
    out.convert('RGB').save(out_path)
    print(f'wrote {out_path}  avg delta = {avg:+.2f} dBm')

    # ── Mean Δ across all APs ──
    # 每個 AP 一張 diff,所有 AP 在每格取平均(nanmean)
    diff_stack = np.stack([eg - mg for mg, eg in zip(m_grids, e_grids)], axis=0)
    diff_mean = np.nanmean(diff_stack, axis=0)
    diff_mean = np.where(valid_mask, diff_mean, np.nan)

    rgba = diff_to_rgba(diff_mean, vmin=-3, vmax=3)
    overlay = Image.fromarray(rgba, 'RGBA')
    out = Image.alpha_composite(base_img, overlay)
    draw = ImageDraw.Draw(out)
    avg = np.nanmean(diff_mean)
    draw.rectangle([(2, 2), (W-2, 38)], fill=(255, 255, 255, 220))
    draw.text((6, 4), 'Overall MEAN-RSSI change  (evening - morning)', fill=(0, 0, 0), font=font)
    draw.text((6, 20), f'mean over top-{len(top)} APs  avg={avg:+.2f} dBm',
              fill=(0, 0, 0), font=font)
    draw.rectangle([(2, H-22), (W-2, H-2)], fill=(255, 255, 255, 220))
    draw.text((6, H-20),
              'blue = average RSSI WEAKER  |  red = average RSSI STRONGER  |  +-3 dBm',
              fill=(0, 0, 0), font=font)
    out_path = os.path.join(args.out_dir, 'overall_diff_mean.png')
    out.convert('RGB').save(out_path)
    print(f'wrote {out_path}  avg delta = {avg:+.2f} dBm')


if __name__ == '__main__':
    main()
