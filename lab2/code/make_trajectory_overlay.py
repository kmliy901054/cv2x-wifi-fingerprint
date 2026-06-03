#!/usr/bin/env python3
"""make_trajectory_overlay — draw all path_ids on the map.

Reads trajectories_slim_combined.csv and overlays every (x,y) point onto
psquare.pgm coloured by its r,g,b column (HSV ring per path).

Outputs:
  trajectories_overlay_all.png         coloured paths on map
  trajectories_overlay_all_white.png   on white background (cleaner print)
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.expanduser(
        '~/ros2_ws/wifi_data/trajectories_slim_combined.csv'))
    ap.add_argument('--map-yaml', default=os.path.expanduser(
        '~/ros2_ws/maps/psquare.yaml'))
    ap.add_argument('--out', default=os.path.expanduser(
        '~/ros2_ws/wifi_data/trajectories_overlay_combined.png'))
    ap.add_argument('--out-white', default=os.path.expanduser(
        '~/ros2_ws/wifi_data/trajectories_overlay_combined_white.png'))
    ap.add_argument('--line-width', type=int, default=2)
    ap.add_argument('--max-step-m', type=float, default=0.8,
                    help='Do not connect points further than this apart (skips AMCL teleports)')
    args = ap.parse_args()

    with open(args.map_yaml) as f:
        meta = yaml.safe_load(f)
    pgm_path = os.path.join(os.path.dirname(args.map_yaml), meta['image'])
    map_img = Image.open(pgm_path).convert('RGB')
    W, H = map_img.size
    res = meta['resolution']
    ox, oy = meta['origin'][0], meta['origin'][1]

    paths = defaultdict(list)
    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pid = int(row['path_id'])
            x = float(row['x']); y = float(row['y'])
            r = float(row['r']); g = float(row['g']); b = float(row['b'])
            # world -> pixel
            px = int((x - ox) / res)
            py = H - int((y - oy) / res)
            paths[pid].append((px, py, r, g, b))

    print(f'loaded {len(paths)} paths, {sum(len(v) for v in paths.values())} points')

    for bg_img, out_path in [
            (map_img.copy(), args.out),
            (Image.new('RGB', (W, H), (255, 255, 255)), args.out_white)]:
        canvas = bg_img.copy()
        draw = ImageDraw.Draw(canvas)
        # if white bg, render map outline only
        if out_path == args.out_white:
            # darken obstacles from map
            map_arr = np.array(map_img.convert('L'))
            obs = map_arr < 100
            canvas_arr = np.array(canvas)
            canvas_arr[obs] = [60, 60, 60]
            canvas = Image.fromarray(canvas_arr)
            draw = ImageDraw.Draw(canvas)
        max_step_px = args.max_step_m / res
        for pid, pts in sorted(paths.items()):
            if len(pts) < 2:
                continue
            r, g, b = pts[0][2], pts[0][3], pts[0][4]
            color = (int(r*255), int(g*255), int(b*255))
            for i in range(len(pts) - 1):
                x1, y1 = pts[i][0], pts[i][1]
                x2, y2 = pts[i+1][0], pts[i+1][1]
                dist_px = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                if dist_px > max_step_px:
                    continue  # skip teleport (AMCL initial pose 拉動 / convergence jump)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=args.line_width)
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        except OSError:
            font = ImageFont.load_default()
        title = f'CV2X Lab 2 — All {len(paths)} trajectory paths (HSV ring, 30s each)'
        draw.rectangle([(2, 2), (W-2, 24)], fill=(255, 255, 255, 220))
        draw.text((6, 4), title, fill=(0, 0, 0), font=font)
        canvas.save(out_path)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
