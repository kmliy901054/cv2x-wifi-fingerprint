"""Real-time WiFi indoor-localization demo (CPU-only, self-contained).

Loads the committed winning model (Cascade 2-level, 5-seed ensemble) and
localizes you live from an ESP32 WiFi scanner, drawing your estimated
position + the model's probability heatmap on the floor plan.

No GPU needed: a full 5-seed ensemble inference is ~9 ms on CPU, negligible
vs the ~3 s an ESP32 WiFi scan takes.

Two input modes
---------------
  Live  (needs an ESP32 on serial + `pip install pyserial`):
      python realtime_demo.py --port COM3            # Windows
      python realtime_demo.py --port /dev/ttyACM0    # Linux

  Replay (no hardware — step through a recorded .jsonl, also shows the
          recorded ground-truth pose so you can eyeball accuracy):
      python realtime_demo.py --replay ../../wifi/wifi_20260517_101315.jsonl

Self-contained: only needs this repo (model weights, bssids.json, map/).
Does NOT require the training dataset to be present.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import data       # only for build_heatmap_grid / build_free_mask / build_fine_to_coarse (map-only)
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / 'outputs' / 'checkpoints'
BSSID_JSON = HERE / 'outputs' / 'bssids.json'
SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
MAX_APS = 50
RSSI_MISSING = -100.0


# ─────────────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────────────
class Localizer:
    """Holds the 5-seed Cascade ensemble + grid; maps a scan → (x,y) + heatmap."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        torch.set_num_threads(max(1, (torch.get_num_threads())))
        self.bssids = json.load(open(BSSID_JSON, encoding='utf-8'))
        self.bidx = {b.upper(): i for i, b in enumerate(self.bssids)}

        # grids (built from the map file only — no training data needed)
        self.fine_xy, self.Gw, self.Gh = data.build_heatmap_grid(cell_size=0.4)
        fine_mask = data.build_free_mask(self.fine_xy)
        coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
        coarse_mask = data.build_free_mask(coarse_xy)
        f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)
        self.bounds = data.HEATMAP_BOUNDS

        self.models = []
        for s in SEEDS:
            ck = CKPT_DIR / f'A_random__Cascade_s{s}.pt'
            if not ck.exists():
                print(f'[warn] missing checkpoint {ck.name} — skipping seed {s}')
                continue
            m = models.SetTransformerHeatmapCascade(
                num_bssids=len(self.bssids),
                fine_cell_xy=self.fine_xy, fine_free_mask=fine_mask.astype(np.float32),
                coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
                fine_to_coarse=f2c, **CFG).to(self.device)
            m.load_state_dict(torch.load(ck, map_location=self.device))
            m.eval()
            self.models.append(m)
        if not self.models:
            raise RuntimeError('No checkpoints found. Did you clone with the '
                               'committed Cascade weights?')
        print(f'[ok] loaded {len(self.models)} seed models on {self.device}')

    def scan_to_input(self, aps):
        """aps: list of {'bssid','rssi'} → (idx, val, mask) torch tensors (1, MAX_APS)."""
        idx = np.full((1, MAX_APS), len(self.bssids), dtype=np.int64)
        val = np.zeros((1, MAX_APS), dtype=np.float32)
        mask = np.zeros((1, MAX_APS), dtype=np.float32)
        ranked = sorted(
            ((str(a.get('bssid', '')).upper(), float(a.get('rssi', RSSI_MISSING)))
             for a in aps if str(a.get('bssid', '')).upper() in self.bidx),
            key=lambda kv: kv[1], reverse=True)[:MAX_APS]
        n_matched = len(ranked)
        for j, (b, v) in enumerate(ranked):
            idx[0, j] = self.bidx[b]
            val[0, j] = (v - RSSI_MISSING) / 20.0
            mask[0, j] = 1.0
        return (torch.from_numpy(idx).to(self.device),
                torch.from_numpy(val).to(self.device),
                torch.from_numpy(mask).to(self.device),
                n_matched)

    @torch.no_grad()
    def localize(self, aps):
        """Return (xy[2], heatmap[Gh,Gw], n_matched_aps)."""
        it, vt, mt, n_matched = self.scan_to_input(aps)
        xys = []
        probs = []
        for m in self.models:
            fine_l, coarse_l = m(it, vt, mt)
            p = m._gated_fine_prob(fine_l, coarse_l)        # (1, Gf)
            probs.append(p.cpu().numpy()[0])
            xys.append((p @ m.fine_xy).cpu().numpy()[0])
        xy = np.mean(xys, axis=0)
        heat = np.mean(probs, axis=0).reshape(self.Gh, self.Gw)
        return xy, heat, n_matched


# ─────────────────────────────────────────────────────────────────────
# Scan sources
# ─────────────────────────────────────────────────────────────────────
def replay_source(path, interval):
    """Yield (aps, true_xy_or_None) from a recorded jsonl, pacing by `interval` s."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            aps = d.get('aps', [])
            pose = d.get('pose') or {}
            true_xy = None
            if pose.get('x') is not None and pose.get('y') is not None:
                true_xy = (float(pose['x']), float(pose['y']))
            yield aps, true_xy
            time.sleep(interval)


def serial_source(port, baud):
    """Yield (aps, None) from a live ESP32 over serial."""
    try:
        import serial
    except ImportError:
        print('[err] pyserial not installed. Run:  pip install pyserial')
        sys.exit(1)
    buf = b''
    while True:
        try:
            with serial.Serial(port, baud, timeout=1.0) as ser:
                print(f'[ok] serial open: {port} @ {baud}')
                while True:
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b'\n' in buf:
                        raw, buf = buf.split(b'\n', 1)
                        s = raw.decode('utf-8', errors='replace').strip()
                        if not s.startswith('{'):
                            continue
                        try:
                            d = json.loads(s)
                        except json.JSONDecodeError:
                            continue
                        yield d.get('aps', []), None
        except Exception as e:                                  # noqa: BLE001
            print(f'[warn] serial error: {e}; retry in 2s')
            time.sleep(2.0)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────
def run_viz(loc, source, show_truth):
    import matplotlib
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image

    # load floor plan
    map_yaml = HERE.parents[1] / 'map' / 'psquare.yaml'
    with open(map_yaml) as f:
        mp = yaml.safe_load(f)
    pgm = np.array(Image.open(map_yaml.parent / mp['image']))
    ox, oy, res = mp['origin'][0], mp['origin'][1], mp['resolution']
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]
    bx0, bx1, by0, by1 = loc.bounds

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(pgm, cmap='gray', origin='lower', extent=extent, alpha=0.7)
    try:
        hot = matplotlib.colormaps['hot'].copy()          # mpl >= 3.6
    except AttributeError:
        hot = matplotlib.cm.get_cmap('hot').copy()         # older mpl
    hot.set_bad(alpha=0.0)            # masked (low-prob) cells render transparent
    heat_im = ax.imshow(np.ma.masked_all((loc.Gh, loc.Gw)), origin='lower',
                         extent=[bx0, bx1, by0, by1], cmap=hot,
                         alpha=0.6, vmin=0, vmax=0.05, zorder=2,
                         interpolation='bilinear')
    trail = deque(maxlen=15)
    (trail_ln,) = ax.plot([], [], '-', color='deepskyblue', lw=1.5,
                           alpha=0.7, zorder=3)
    (pred_pt,) = ax.plot([], [], 'o', color='deepskyblue', ms=16,
                          mec='black', mew=1.5, zorder=5, label='predicted')
    (true_pt,) = ax.plot([], [], 'X', color='lime', ms=14, mec='black',
                          mew=1.5, zorder=4, label='true (replay)')
    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.set_xlim(-7, 12); ax.set_ylim(-3, 14); ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('WiFi Indoor Localization — live demo (Cascade 5-seed)')
    ax.legend(loc='upper right')

    n = 0
    err_hist = []
    for aps, true_xy in source:
        t0 = time.time()
        xy, heat, n_matched = loc.localize(aps)
        dt = (time.time() - t0) * 1000
        n += 1

        heat_im.set_data(np.ma.masked_less(heat, max(1e-4, heat.max() * 0.08)))
        heat_im.set_clim(0, max(0.02, heat.max()))
        trail.append(xy)
        tr = np.array(trail)
        trail_ln.set_data(tr[:, 0], tr[:, 1])
        pred_pt.set_data([xy[0]], [xy[1]])

        info = [f'scan #{n}',
                f'APs matched: {n_matched}/{len(aps)}',
                f'pred: ({xy[0]:+.2f}, {xy[1]:+.2f}) m',
                f'infer: {dt:.0f} ms']
        if show_truth and true_xy is not None:
            true_pt.set_data([true_xy[0]], [true_xy[1]])
            e = float(np.hypot(xy[0] - true_xy[0], xy[1] - true_xy[1]))
            err_hist.append(e)
            info.append(f'true: ({true_xy[0]:+.2f}, {true_xy[1]:+.2f}) m')
            info.append(f'error: {e:.2f} m   (median {np.median(err_hist):.2f})')
        txt.set_text('\n'.join(info))
        plt.pause(0.001)

    if err_hist:
        print(f'\nReplay finished: {n} scans, '
              f'median error {np.median(err_hist):.3f} m, '
              f'mean {np.mean(err_hist):.3f} m')
    plt.ioff()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Real-time WiFi localization demo')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--replay', type=str, help='path to a recorded .jsonl to replay')
    g.add_argument('--port', type=str, help='serial port of the ESP32 (live)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--interval', type=float, default=0.6,
                    help='replay seconds between scans (default 0.6)')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--no-viz', action='store_true',
                    help='headless: print predictions, no matplotlib window')
    args = ap.parse_args()

    loc = Localizer(device=args.device)

    if args.replay:
        src = replay_source(args.replay, args.interval)
        show_truth = True
    else:
        src = serial_source(args.port, args.baud)
        show_truth = False

    if args.no_viz:
        errs = []
        for n, (aps, true_xy) in enumerate(src, 1):
            xy, _, nm = loc.localize(aps)
            line = f'#{n:4d}  pred=({xy[0]:+.2f},{xy[1]:+.2f})  APs={nm}'
            if true_xy is not None:
                e = float(np.hypot(xy[0] - true_xy[0], xy[1] - true_xy[1]))
                errs.append(e)
                line += f'  true=({true_xy[0]:+.2f},{true_xy[1]:+.2f})  err={e:.2f}m'
            print(line, flush=True)
        if errs:
            print(f'\nmedian error {np.median(errs):.3f} m over {len(errs)} scans')
    else:
        run_viz(loc, src, show_truth)


if __name__ == '__main__':
    main()
