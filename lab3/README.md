# Lab 3 — WiFi Indoor Localization with Deep Learning

Uses the Lab 2 fingerprint dataset (1,812 (RSSI, pose) records over a 189.5 m²
lab) to train an indoor-localization model. The journey goes from a classic KNN
baseline (median 1.57 m) to a coarse-to-fine cascade (median **0.79 m**), then
runs live on an ESP32 in the lab.

## Read in this order

| Doc | What it covers |
|---|---|
| [EVOLUTION.md](EVOLUTION.md) | The full model evolution story, 1.57 m → 0.79 m, including dead ends |
| [LAB3_REPORT.md](LAB3_REPORT.md) | The course report (problem / method / splits / results) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model details + training recipes |
| [DEMO.md](DEMO.md) | How to run the real-time demo (matplotlib or ROS 2 + RViz) |
| [outputs/slides/](outputs/slides/) | The presentation deck (`lab3_journey.pptx`) + how it was built |

## Result (Split A random 80/20 test, 363 samples)

| Model | Median | Notes |
|---|---:|---|
| KNN k=5 | 1.568 m | classical fingerprinting baseline |
| Set Transformer MDN | 1.093 m | variable-length set input |
| + GP synthetic data | 0.906 m | fills spatial coverage gaps |
| Heatmap + free-mask (×5-ens) | 0.883 m | grid classification, not regression |
| **Cascade ×5-ens** | **0.793 m** | coarse gate over fine grid — winner |

Reproduce the headline number from the committed weights:
```bash
python3 load_best_model.py     # → median 0.793 m on Split A test
```

## Code layout

```
data.py, models.py            data pipeline + all model definitions
train_*.py                    one trainer per experiment in the ladder
load_best_model.py            reproduce the 0.793 m result from committed weights
evaluate.py, tta.py           evaluation + test-time augmentation
synthetic.py                  GP-kriging synthetic data generation

esp32_localizer_ros.py        real-time ROS 2 node (live inference)
lab3_demo.launch.py           launches node + RViz
esp32_localizer.rviz          RViz layout
run_at_lab.sh                 one-command live/replay wrapper
gt_error_node.py              click ground truth in RViz, print live error
realtime_demo.py              matplotlib version of the live demo

make_report_figures.py        result figures (CDF, ladder, scatter, reliability, ...)
make_arch_figures.py          per-model architecture block diagrams
make_gating_figure.py         the "why the cascade wins" gating figure

outputs/
  checkpoints/                model weights (only the winning 5-seed cascade is committed)
  predictions/                per-model test predictions (.npz) for all splits
  figures/                    report + slide figures
  plots/, plots_synthetic/    EDA + synthetic-data diagnostics
  slides/                     the presentation deck
  metrics.csv                 every experiment's metrics
```

## Real-time demo, in one line

```bash
sudo chmod 666 /dev/ttyACM0    # once per boot
./run_at_lab.sh                # live ESP32 + RViz; or `./run_at_lab.sh replay`
```

See [DEMO.md](DEMO.md) for details and troubleshooting.
