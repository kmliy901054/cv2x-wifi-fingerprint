# CV2X Lab 2 — WiFi Fingerprint Indoor Localization

TurtleBot3 + LDS-01 + ESP32-S3 室內 WiFi fingerprint dataset 蒐集系統。

NYCU 國立陽明交通大學 CV2X 課程 Lab 2 完整交付,**221 條軌跡 / 1,812 筆 WiFi-pose / 50,196 個 AP detection / 115 個 unique BSSID**,涵蓋 189.5 m²,含 52 個 AP 的早晚對比分析。

> 📘 完整技術報告:[`REPORT.md`](REPORT.md)  • 🛠 重現指南:[`SETUP.md`](SETUP.md)  • 🎤 報告講稿:[`PRESENTATION.md`](PRESENTATION.md)

---

## ✨ 結果速覽

| 全部 221 條軌跡疊在地圖上 | RSSI heatmap(combined-best) |
|:---:|:---:|
| ![trajectories](visualizations/trajectories_overlay_combined.png) | ![heatmap](visualizations/heatmap_combined_best.png) |
| 哪台 AP 在每格主導 | 早晚 RSSI 差異(BMELab)|
| ![dominant](visualizations/heatmap_dominant_ap.png) | ![diff](visualizations/morning_vs_evening/diff_03_BMELab.png) |

---

## 🏗 系統架構

| SLAM 階段 ROS node graph | AMCL 蒐集階段 ROS node graph |
|:---:|:---:|
| ![slam](architecture/rqt_graph_slam.png) | ![amcl](architecture/rqt_graph_amcl.png) |

完整 TF tree 與 node 分工說明見 [`architecture/NODE_ARCHITECTURE.md`](architecture/NODE_ARCHITECTURE.md)。

---

## 📦 目錄結構

```
.
├── REPORT.md                完整技術報告(458 行,9 章)
├── README.md                本檔
├── SETUP.md                 從零重現指南
├── PRESENTATION.md          報告講稿 / slide outline
├── LICENSE                  MIT
├── map/                     SLAM 地圖(psquare.pgm + .yaml)
├── trajectories/            slim/wide CSV(1,513 / 50,196 列)
├── wifi/                    原始 jsonl × 6
├── visualizations/          軌跡疊圖、heatmap、早晚 diff
├── architecture/            rqt_graph、TF tree、node 分工 md
└── code/                    所有 Python / launch / msg / yaml
```

---

## 🔑 核心設計亮點

| 設計 | 為何 |
|---|---|
| **AMCL 純定位**(蒐集時不再 SLAM)| 確保所有 wifi record reference 同一張地圖,fingerprint 一致 |
| **Event-driven wifi-pose 配對** | 符合 spec「有新的 WiFi 結果就紀錄座標與 WiFi」,非 timer-driven |
| **TF lookup at `scan stamp`** | 修了 spec 警告過的 bug(原版用 latest TF → 配到 3 秒前 wifi 是當下 pose,1 m 偏移)|
| **ESP32 backdate stamp** | `now − scan_dur/2`,進一步降低 wifi-pose 偏移到 < 0.1 秒 |
| **30 秒切片**(T 可調)| `--split-by-time T` CLI,直接對應 spec「軌跡為 T 秒,可設定參數」 |
| **CycloneDDS RMW** | FastRTPS over WiFi 會吞 `/tf` → 跨機 SLAM 必死 |
| **previous_paths visualizer** | 收第二輪時,RViz 顯示早上走過的彩色 path → 不重複走 |
| **早晚 dataset 對比** | 52 個 AP 跨時段 ±2.6 dBm 內,量化「人體吸收效應」(實驗室晚上 AP 變強 +2.6 dBm)|

---

## 🚀 Quick start

```bash
# 建圖
ros2 launch code/lab2.launch.py

# 蒐集 wifi(已有地圖)
ros2 launch code/lab2_amcl.launch.py

# 後處理
cd code
python3 jsonl_to_csv.py --split-by-time 30
python3 make_rssi_heatmap.py
python3 make_morning_evening_diff.py
python3 make_trajectory_overlay.py
```

詳細步驟見 [`SETUP.md`](SETUP.md)。

---

## 📊 蒐集統計

| 指標 | 早上 (5/17) | 晚上 (5/23) | 合計 |
|---|---:|---:|---:|
| WiFi-pose record | 912 | 900 | **1,812** |
| AP detection | 24,359 | 25,837 | **50,196** |
| 路徑數(30s)| 113 | 108 | **221** |
| 平均 AP / scan | 26.7 | 28.7 | **27.7** |
| Unique BSSID | 89 | 102 | **115** |

涵蓋 **15.97 × 11.87 m = 189.5 m²** bbox。

---

## 📜 License

MIT — 見 [LICENSE](LICENSE)。

## 🙏 Acknowledgements

Cartographer(Google),nav2(Steve Macenski),TurtleBot3(ROBOTIS),CycloneDDS(Eclipse)。
本實驗開發協助:Claude(Anthropic)。
