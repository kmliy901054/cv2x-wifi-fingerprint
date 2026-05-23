#!/usr/bin/env python3
"""jsonl_to_csv — merge wifi JSONL + path_metadata.json into senior's CSV format.

Senior's CSV format:
  path_id, x, y, z, r, g, b

We extend it with WiFi info for fingerprinting:
  path_id, x, y, z, r, g, b, yaw, ssid, bssid, rssi, channel, encryption

For each WifiScan record we emit ONE row per AP visible at that scan, sharing
the same (path_id, x, y, z, r, g, b, yaw).

Usage:
  python3 ~/ros2_ws/lab2_configs/jsonl_to_csv.py \\
      --wifi-dir ~/ros2_ws/wifi_data \\
      --metadata ~/ros2_ws/wifi_data/path_metadata.json \\
      --output ~/ros2_ws/wifi_data/trajectories.csv \\
      --slim-output ~/ros2_ws/wifi_data/trajectories_slim.csv

  # Skip wifi columns, output only trajectory pose CSV (matches senior exactly):
  python3 jsonl_to_csv.py --slim-only
"""
import argparse
import colorsys
import csv
import glob
import json
import os
import sys


def load_metadata(path):
    if not os.path.exists(path):
        print(f'[warn] no metadata at {path} — every row will have path_id=0')
        return {'paths': {}}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_path_id(wall_time: float, metadata: dict) -> tuple:
    """Map wall_time to path_id + color via metadata.paths intervals."""
    for pid_str, info in metadata.get('paths', {}).items():
        s = info.get('start_wall')
        e = info.get('end_wall')
        if s is None: continue
        if e is None: e = float('inf')
        if s <= wall_time <= e:
            return int(pid_str), info.get('color_rgb', [0.5, 0.5, 0.5])
    return 0, [0.5, 0.5, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wifi-dir', default=os.path.expanduser('~/ros2_ws/wifi_data'))
    ap.add_argument('--metadata', default=os.path.expanduser('~/ros2_ws/wifi_data/path_metadata.json'))
    ap.add_argument('--output', default=os.path.expanduser('~/ros2_ws/wifi_data/trajectories.csv'))
    ap.add_argument('--slim-output', default=os.path.expanduser('~/ros2_ws/wifi_data/trajectories_slim.csv'),
                    help='CSV matching senior format exactly (no wifi cols)')
    ap.add_argument('--slim-only', action='store_true',
                    help='Only emit slim CSV (skip the wide one with wifi cols)')
    ap.add_argument('--per-file-pathid', action='store_true',
                    help='Assign one path_id per jsonl file (when no metadata). '
                         'Each file gets a distinct HSV color too.')
    ap.add_argument('--split-by-time', type=float, default=0,
                    help='Slice each jsonl into chunks of N seconds; each chunk = new path_id. '
                         'Matches lab spec "軌跡為 T 秒".  e.g. --split-by-time 30')
    ap.add_argument('--split-by-distance', type=float, default=0,
                    help='Start new path_id when robot moves > N meters from path start. '
                         'Alternative to time-based split.')
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)
    jsonl_files = sorted(glob.glob(os.path.join(args.wifi_dir, 'wifi_*.jsonl')))
    # Drop empty files
    jsonl_files = [f for f in jsonl_files if os.path.getsize(f) > 0]

    # If --per-file-pathid: build a synthetic metadata mapping each file to its
    # own path_id (1..N) with a distinct HSV color.
    file_path_map = {}   # filepath -> (path_id, [r,g,b])
    if args.per_file_pathid:
        n_files = len(jsonl_files)
        for i, jf in enumerate(jsonl_files, start=1):
            h = (i - 1) / max(1, n_files)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            file_path_map[jf] = (i, [round(r, 3), round(g, 3), round(b, 3)])
        print(f'[per-file-pathid] {n_files} files → path_ids 1..{n_files}')

    # --split-by-time / --split-by-distance: pre-compute per-record path_id
    # by chunking each file's records.  Returns dict: (filepath, line_idx) -> (pid, rgb)
    record_path_map = {}
    if args.split_by_time > 0 or args.split_by_distance > 0:
        all_chunks = []   # list of (file, [line_indices])
        for jf in jsonl_files:
            with open(jf, 'r', encoding='utf-8') as f:
                records = []
                for li, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try: rec = json.loads(line)
                    except: continue
                    records.append((li, rec))
            if not records: continue
            cur_chunk = [records[0][0]]
            base_t = records[0][1].get('wall_time', 0)
            base_xy = (records[0][1].get('pose', {}).get('x', 0),
                        records[0][1].get('pose', {}).get('y', 0))
            for li, rec in records[1:]:
                t = rec.get('wall_time', 0)
                x = rec.get('pose', {}).get('x', 0)
                y = rec.get('pose', {}).get('y', 0)
                start_new = False
                if args.split_by_time > 0 and (t - base_t) >= args.split_by_time:
                    start_new = True
                if args.split_by_distance > 0:
                    d = ((x - base_xy[0])**2 + (y - base_xy[1])**2) ** 0.5
                    if d >= args.split_by_distance:
                        start_new = True
                if start_new:
                    all_chunks.append((jf, cur_chunk))
                    cur_chunk = [li]
                    base_t, base_xy = t, (x, y)
                else:
                    cur_chunk.append(li)
            if cur_chunk:
                all_chunks.append((jf, cur_chunk))

        n_chunks = len(all_chunks)
        for i, (jf, lis) in enumerate(all_chunks, start=1):
            h = (i - 1) / max(1, n_chunks)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            color = [round(r, 3), round(g, 3), round(b, 3)]
            for li in lis:
                record_path_map[(jf, li)] = (i, color)
        mode = 'time' if args.split_by_time > 0 else 'distance'
        print(f'[split-by-{mode}] sliced {len(jsonl_files)} files into {n_chunks} path chunks')
    if not jsonl_files:
        print(f'[error] no wifi_*.jsonl in {args.wifi_dir}'); sys.exit(1)
    print(f'reading {len(jsonl_files)} jsonl files...')

    # Sets to dedup rows in slim CSV (one row per (path_id, x, y))
    slim_seen = set()
    wide_rows = 0
    slim_rows = 0

    wide_f = None if args.slim_only else open(args.output, 'w', newline='', encoding='utf-8')
    slim_f = open(args.slim_output, 'w', newline='', encoding='utf-8')

    wide = csv.writer(wide_f) if wide_f else None
    slim = csv.writer(slim_f)
    if wide:
        wide.writerow(['path_id','x','y','z','r','g','b','yaw',
                        'ssid','bssid','rssi','channel','encryption',
                        'stamp_sec','stamp_nanosec'])
    slim.writerow(['path_id','x','y','z','r','g','b'])

    for jf in jsonl_files:
        with open(jf, 'r', encoding='utf-8') as f:
            for li, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pose = rec.get('pose') or {}
                x = float(pose.get('x', 0.0))
                y = float(pose.get('y', 0.0))
                z = float(pose.get('z', 0.0))
                yaw = float(pose.get('yaw', 0.0))
                # priority: per-record (split) > per-file > time-metadata
                if (jf, li) in record_path_map:
                    pid, rgb = record_path_map[(jf, li)]
                elif jf in file_path_map:
                    pid, rgb = file_path_map[jf]
                else:
                    pid, rgb = find_path_id(rec.get('wall_time', 0), metadata)
                r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

                # Slim: one row per (path_id, x, y) — dedup
                key = (pid, round(x, 3), round(y, 3))
                if key not in slim_seen:
                    slim.writerow([pid, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}',
                                    int(round(r)), int(round(g)), int(round(b))])
                    slim_seen.add(key)
                    slim_rows += 1

                # Wide: one row per (scan, AP)
                if wide:
                    stamp = rec.get('stamp', {})
                    for ap_rec in rec.get('aps', []):
                        wide.writerow([
                            pid, f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
                            f'{r:.3f}', f'{g:.3f}', f'{b:.3f}', f'{yaw:.4f}',
                            ap_rec.get('ssid',''),
                            ap_rec.get('bssid',''),
                            ap_rec.get('rssi',0),
                            ap_rec.get('channel',0),
                            ap_rec.get('encryption',''),
                            stamp.get('sec',0),
                            stamp.get('nanosec',0),
                        ])
                        wide_rows += 1

    if wide_f: wide_f.close()
    slim_f.close()

    print(f'✓ slim CSV (senior format): {slim_rows:,} rows → {args.slim_output}')
    if not args.slim_only:
        print(f'✓ wide CSV (with wifi):     {wide_rows:,} rows → {args.output}')
    # Some stats
    pids = set()
    for k in slim_seen: pids.add(k[0])
    print(f'  distinct path_ids: {sorted(pids)}')


if __name__ == '__main__':
    main()
