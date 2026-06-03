"""分早上、晚上、合併三張軌跡圖。"""
import csv
import os
import json
import glob
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np


def load_map(path):
    with open(path) as f:
        meta = yaml.safe_load(f)
    img = Image.open(os.path.join(os.path.dirname(path), meta['image'])).convert('RGB')
    return img, meta['resolution'], meta['origin'][0], meta['origin'][1]


def world_to_px(x, y, ox, oy, res, H):
    return int((x - ox) / res), H - int((y - oy) / res)


def draw_overlay(map_img, points_by_pid, title, out_path,
                 line_width=2, max_step_px=16):
    W, H = map_img.size
    canvas = map_img.copy()
    draw = ImageDraw.Draw(canvas)
    for pid, pts in sorted(points_by_pid.items()):
        if len(pts) < 2:
            continue
        r, g, b = pts[0][2], pts[0][3], pts[0][4]
        color = (int(r*255), int(g*255), int(b*255))
        for i in range(len(pts) - 1):
            x1, y1 = pts[i][0], pts[i][1]
            x2, y2 = pts[i+1][0], pts[i+1][1]
            if ((x2-x1)**2 + (y2-y1)**2) ** 0.5 > max_step_px:
                continue
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([(2, 2), (W-2, 24)], fill=(255, 255, 255, 220))
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    canvas.save(out_path)
    print(f'wrote {out_path}  ({len(points_by_pid)} paths,'
          f' {sum(len(v) for v in points_by_pid.values())} points)')


def jsonl_to_points(files, map_path, color_mode='hsv'):
    """Read jsonl directly, slice into 30s chunks, build (pid -> list)."""
    import colorsys
    pts = defaultdict(list)
    pid = 0
    map_img, res, ox, oy = load_map(map_path)
    W, H = map_img.size
    for f in sorted(files):
        chunk_start = None
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = d.get('pose', {})
            if p.get('frame_id') != 'map':
                continue
            stamp = d['stamp']['sec'] + d['stamp']['nanosec'] / 1e9 \
                if isinstance(d.get('stamp'), dict) else d.get('stamp', 0)
            if chunk_start is None or stamp - chunk_start > 30.0:
                pid += 1
                chunk_start = stamp
            px, py = world_to_px(p['x'], p['y'], ox, oy, res, H)
            h = (pid * 137.508) % 360 / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            pts[pid].append((px, py, r, g, b))
    return pts, map_img


def main():
    wifi_dir = os.path.expanduser('~/ros2_ws/wifi_data')
    map_yaml = os.path.expanduser('~/ros2_ws/maps/psquare.yaml')
    out_dir = wifi_dir

    morning_files = sorted(glob.glob(os.path.join(wifi_dir, 'wifi_20260517_*.jsonl')))
    evening_files = sorted(glob.glob(os.path.join(wifi_dir, 'wifi_20260523_*.jsonl')))
    morning_files = [f for f in morning_files if os.path.getsize(f) > 0]
    evening_files = [f for f in evening_files if os.path.getsize(f) > 0]

    print(f'morning: {len(morning_files)} files, evening: {len(evening_files)} files')

    m_pts, map_img = jsonl_to_points(morning_files, map_yaml)
    e_pts, _ = jsonl_to_points(evening_files, map_yaml)

    draw_overlay(map_img, m_pts,
        f'Morning session (2026-05-17) — {len(m_pts)} paths, {sum(len(v) for v in m_pts.values())} pts',
        os.path.join(out_dir, 'trajectories_overlay_morning.png'))
    draw_overlay(map_img, e_pts,
        f'Evening session (2026-05-23) — {len(e_pts)} paths, {sum(len(v) for v in e_pts.values())} pts',
        os.path.join(out_dir, 'trajectories_overlay_evening.png'))


if __name__ == '__main__':
    main()
