---
name: run-lab3
description: Run, launch, smoke-test, reproduce, or screenshot the Lab 3 WiFi indoor-localization model (Set Transformer + cascade). Use when asked to run lab3, reproduce the 0.76 m localization result, run/launch the realtime demo, or screenshot the live localization on the floor plan.
---

# Run Lab 3 — WiFi indoor localization

Deep-learning indoor localization: one WiFi scan (a variable-length set of
`(BSSID, RSSI)`) → robot `(x, y)` on the lab floor plan. Champion model is a
Set Transformer encoder + coarse-to-fine heatmap cascade, trained on 1,449 real
+ 5,000 GP-synthetic scans, 5-seed geometric-median ensemble.

It is driven by **`.claude/skills/run-lab3/driver.py`** — one handle that checks
the environment, reproduces the headline metric from committed weights, runs the
real realtime-demo CLI in replay mode, and renders a live-demo frame to a PNG.
**CPU-only is fine** (~9 ms/scan); no GPU and no ESP32 hardware required.

All paths below are relative to `lab3/` (the unit). Use the Anaconda interpreter
— it is the one with `torch`+CUDA and sklearn (see Gotchas).

## Prerequisites

Committed weights mean there is **no build step**. Just the Python deps:

```bash
pip install torch numpy scipy scikit-learn matplotlib pyyaml pillow pyserial
```

(`pyserial` is only needed for live ESP32 mode; everything else runs without it.
A CPU-only `torch` wheel is sufficient.)

## Run (agent path) — the driver

```bash
# from lab3/
/c/ProgramData/anaconda3/python.exe .claude/skills/run-lab3/driver.py smoke
```

`smoke` runs all four steps and prints `PASS smoke: all green`:

- `check` — deps + the 5 `CascadeTuned` checkpoints + replay data + map present
- `reproduce` — loads committed weights → `Split A test median = 0.760 m`
- `demo` — runs the real `realtime_demo.py` CLI in replay mode (58 scans localized)
- `screenshot` — renders one live frame (floor plan + probability heatmap +
  predicted vs true) to `.claude/skills/run-lab3/demo_screenshot.png`

Run any step alone, e.g.:

```bash
/c/ProgramData/anaconda3/python.exe .claude/skills/run-lab3/driver.py screenshot
```

The screenshot is the live-demo GUI surface rendered headless (Agg) — open
`demo_screenshot.png` to see the predicted point sitting on the ground truth.

## Direct invocation (no driver)

Reproduce a result number straight from committed weights:

```bash
# from lab3/  — honest headline (Cascade-tuned 5-seed ensemble)
/c/ProgramData/anaconda3/python.exe load_best_model.py --variant tuned     # median 0.760 m
/c/ProgramData/anaconda3/python.exe load_best_model.py --variant baseline  # median 0.793 m
```

Drive the actual app (replay mode, no hardware, headless):

```bash
# from lab3/
/c/ProgramData/anaconda3/python.exe realtime_demo.py \
  --replay ../wifi/wifi_20260517_101315.jsonl --no-viz --interval 0
```

## Run (human path) — live + windowed

With the **real ESP32** on serial (drops you into the matplotlib live view):

```bash
# Windows COM port / Linux tty — needs pyserial + a display
python realtime_demo.py --port COM3 --smooth 5
```

Or replay with the live window (no `--no-viz`): a matplotlib window animates the
predicted dot + heatmap. Headless this is useless (the window never appears) —
use the driver's `screenshot` step instead.

## Gotchas (battle scars)

- **Use `/c/ProgramData/anaconda3/python.exe`, not `python`.** The bare `python`
  on PATH is an msys build with **no numpy** (`ModuleNotFoundError: numpy`); the
  Anaconda interpreter is the one with torch+CUDA, sklearn, etc.
- **Project scripts must run from `lab3/`** — they do `import data/models/synthetic`.
  The driver sets `cwd=lab3` and adds it to `sys.path`, so the driver works from
  anywhere; direct invocations must `cd lab3/` first.
- **`0.760` is the honest number; `0.650` is NOT.** `load_best_model.py` defaults
  to the a-priori `tuned` config (0.760 m). A separate `load_greedy_winner.py`
  prints 0.650 m but that was test-set-tuned (overfit); don't quote it as the
  result. The leak-free nested-CV number is ~0.94 m (`honest_nested_cv.py`).
- **Replay errors look optimistic.** The bundled `wifi_*.jsonl` were part of
  training (Split A is random 80/20), so replayed per-scan errors (~0.1 m) flatter
  the model. The honest figure is `load_best_model.py`'s test median.
- **Headless matplotlib.** `realtime_demo.py` without `--no-viz` calls
  `plt.ion()/plt.show()`; headless that prints a harmless backend no-op and exits.
  For a headless visual, use `driver.py screenshot` (forces the Agg backend).
- **GP-synth scripts spam `ConvergenceWarning`.** The training / honest-validation
  scripts (not needed to *run* the model) flood stdout with sklearn GP warnings;
  filter with `grep -v "ConvergenceWarning\|warnings.warn\|kernels.py"`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | You used the wrong Python; use `/c/ProgramData/anaconda3/python.exe` |
| `No checkpoints found` / check fails | Ensure committed `outputs/checkpoints/A_random__CascadeTuned_s4*.pt` are present (they're in git) |
| `ModuleNotFoundError: serial` | Only needed for `--port` live mode: `pip install pyserial`; replay mode doesn't need it |
| Demo shows `APs matched: 0/N` | The scan's BSSIDs aren't in the trained 80-AP vocabulary (wrong/old data); the model only knows Lab 2's APs |
