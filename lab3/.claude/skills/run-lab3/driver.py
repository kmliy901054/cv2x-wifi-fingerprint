#!/usr/bin/env python3
"""Driver for the Lab 3 WiFi indoor-localization model.

One handle to build/verify/drive the project from a clean machine:

  python driver.py check        # deps + committed checkpoints + data present
  python driver.py reproduce    # load committed weights -> Split A median (~0.76 m)
  python driver.py demo         # run the REAL realtime_demo CLI in replay mode
  python driver.py screenshot   # render one live-demo frame to a PNG (the GUI surface)
  python driver.py smoke        # all of the above (default)

CPU-only is fine (inference ~9 ms/scan). No ESP32 hardware needed: the demo's
replay mode feeds recorded scans through the exact live pipeline.

The skill lives at  lab3/.claude/skills/run-lab3/ , so the project root (lab3/)
is parents[3] of this file. We add it to sys.path and run the project's scripts
with cwd=lab3 so their `import data/models/synthetic` resolve.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB3 = HERE.parents[2]              # run-lab3 -> skills -> .claude -> lab3
REPO = LAB3.parent
PY = sys.executable
REPLAY = REPO / 'wifi' / 'wifi_20260517_101315.jsonl'
SHOT = HERE / 'demo_screenshot.png'

OK = '\033[92mPASS\033[0m'
NO = '\033[91mFAIL\033[0m'


def run(cmd, **kw):
    print(f'$ {" ".join(str(c) for c in cmd)}')
    return subprocess.run(cmd, cwd=str(LAB3), text=True,
                          capture_output=True, **kw)


def cmd_check():
    import importlib
    mods = ['torch', 'numpy', 'scipy', 'sklearn', 'matplotlib', 'yaml', 'PIL']
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    assert not missing, f'missing python deps: {missing}'
    ck = sorted((LAB3 / 'outputs' / 'checkpoints').glob('A_random__CascadeTuned_s4*.pt'))
    assert len(ck) >= 5, f'expected 5 CascadeTuned checkpoints, found {len(ck)}'
    assert REPLAY.exists(), f'missing replay data: {REPLAY}'
    assert (REPO / 'map' / 'psquare.yaml').exists(), 'missing map/psquare.yaml'
    import torch
    print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}')
    print(f'  {len(ck)} checkpoints, replay data + map present')
    print(OK, 'check')


def cmd_reproduce():
    r = run([PY, 'load_best_model.py', '--variant', 'tuned'])
    sys.stdout.write(r.stdout[-600:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-800:]); raise SystemExit('reproduce failed')
    med = None
    for line in r.stdout.splitlines():
        if 'median' in line and '=' in line:
            try:
                med = float(line.split('=')[1].split('m')[0])
            except Exception:
                pass
    assert med is not None, 'could not parse median from output'
    assert med < 0.85, f'median {med} unexpectedly high'
    print(f'{OK} reproduce  (Split A test median = {med:.3f} m)')
    return med


def cmd_demo():
    r = run([PY, 'realtime_demo.py', '--replay', str(REPLAY),
             '--no-viz', '--interval', '0'])
    tail = [l for l in r.stdout.splitlines() if l.startswith('#')]
    sys.stdout.write('\n'.join(tail[:5]) + '\n')
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-800:]); raise SystemExit('demo failed')
    assert any('err=' in l for l in tail), 'demo produced no predictions'
    assert 'median error' in r.stdout, 'demo did not summarize'
    print(f'{OK} demo  ({len(tail)} replayed scans localized)')


def cmd_screenshot():
    """Render one live-demo frame (floor plan + probability heatmap + predicted
    point) to a PNG — the GUI surface, headless via the Agg backend."""
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(LAB3))
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np, yaml
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import realtime_demo as rd

    loc = rd.Localizer(device='cpu')
    scan = None
    for line in open(REPLAY, encoding='utf-8'):
        line = line.strip()
        if line.startswith('{'):
            d = json.loads(line)
            if d.get('aps'):
                scan = d; break
    assert scan is not None, 'no scan in replay file'
    xy, heat, n = loc.localize(scan['aps'])
    true = (scan.get('pose') or {}).get('x'), (scan.get('pose') or {}).get('y')

    my = REPO / 'map' / 'psquare.yaml'
    mp = yaml.safe_load(open(my)); pgm = np.array(Image.open(my.parent / mp['image']))
    ox, oy, res = mp['origin'][0], mp['origin'][1], mp['resolution']
    H, W = pgm.shape
    extent = [ox, ox + W * res, oy, oy + H * res]
    bx0, bx1, by0, by1 = loc.bounds
    hot = matplotlib.colormaps['hot'].copy(); hot.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pgm, cmap='gray', origin='lower', extent=extent, alpha=0.7)
    ax.imshow(np.ma.masked_less(heat, heat.max() * 0.08), origin='lower',
              extent=[bx0, bx1, by0, by1], cmap=hot, alpha=0.6,
              vmin=0, vmax=heat.max(), zorder=2, interpolation='bilinear')
    ax.plot([xy[0]], [xy[1]], 'o', color='deepskyblue', ms=16, mec='black',
            mew=1.5, zorder=5, label='predicted')
    if true[0] is not None:
        ax.plot([true[0]], [true[1]], 'X', color='lime', ms=14, mec='black',
                mew=1.5, zorder=4, label='true')
    ax.set_xlim(-7, 12); ax.set_ylim(-3, 14); ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'run-lab3 driver: live localize ({n} APs matched)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.tight_layout(); fig.savefig(SHOT, dpi=120); plt.close(fig)
    assert SHOT.exists() and SHOT.stat().st_size > 5000, 'screenshot not written'
    print(f'{OK} screenshot  -> {SHOT}  ({SHOT.stat().st_size//1024} KB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', default='smoke',
                    choices=['check', 'reproduce', 'demo', 'screenshot', 'smoke'])
    a = ap.parse_args()
    if a.cmd == 'smoke':
        cmd_check(); cmd_reproduce(); cmd_demo(); cmd_screenshot()
        print('\n' + OK + ' smoke: all green')
    else:
        globals()[f'cmd_{a.cmd}']()


if __name__ == '__main__':
    main()
