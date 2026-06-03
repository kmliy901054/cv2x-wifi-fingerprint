# CV2X — WiFi 室內定位(資料蒐集 + 深度學習)

NYCU CV2X 課程專案,分兩階段:

- **Lab 2 — 蒐集**:TurtleBot3 burger + LDS-01 + ESP32-S3,在 BME Lab(189.5 m²)
  邊定位邊掃 WiFi,做出一份 1,812 筆的 (RSSI, pose) 指紋資料集。
- **Lab 3 — 使用**:用這份資料訓練室內定位模型,從 KNN baseline 的中位誤差
  1.57 m 一路做到 coarse-to-fine cascade 的 **0.79 m**,並在實驗室即時 demo。

## 專案結構

```
.
├── lab2/                ROS 蒐集程式、架構圖、視覺化、報告、簡報
├── lab3/                深度學習模型、即時 demo、圖表、簡報
├── map/                 SLAM 樓層平面圖 ── 共用
├── wifi/                原始指紋 jsonl ── 共用(lab2 產、lab3 讀)
├── trajectories/        表格化軌跡 CSV ── 共用
├── README.md            (本檔)
└── LICENSE
```

每個資料夾都有自己的 README 說明內容與格式。

## 兩個 Lab

| | Lab 2 — 蒐集 | Lab 3 — 定位 |
|---|---|---|
| 做什麼 | 開機器人收 WiFi 指紋資料集 | 用資料集訓練定位模型 |
| 入口 | [lab2/README.md](lab2/README.md) | [lab3/README.md](lab3/README.md) |
| 報告 | [lab2/REPORT.md](lab2/REPORT.md) | [lab3/LAB3_REPORT.md](lab3/LAB3_REPORT.md) |
| 故事 | [lab2/PRESENTATION.md](lab2/PRESENTATION.md) | [lab3/EVOLUTION.md](lab3/EVOLUTION.md)(演進史) |
| 簡報 | [pptx](lab2/CV2X_Lab2_presentation.pptx)(27 張) | [lab3_journey.pptx](lab3/outputs/slides/lab3_journey.pptx)(18 張) |

## 資料集(共用,在根目錄)

| | 早 (5/17) | 晚 (5/23) | 合計 |
|---|---:|---:|---:|
| WiFi-pose record | 912 | 900 | 1,812 |
| AP detection | 24,359 | 25,837 | 50,196 |
| 軌跡段 (30s 切) | 113 | 108 | 221 |
| Unique BSSID | 89 | 102 | 115(取 ≥10 次的 80 個當特徵) |

bbox 15.97 × 11.87 m = 189.5 m²。格式說明見
[`wifi/`](wifi/)、[`trajectories/`](trajectories/)、[`map/`](map/)。

## Lab 3 成果(Split A 隨機 80/20 測試,363 筆)

| 模型 | 中位誤差 | 重點 |
|---|---:|---|
| KNN k=5 | 1.568 m | 經典 fingerprinting 基準線 |
| Set Transformer MDN | 1.093 m | 變長集合輸入 |
| + GP 合成資料 | 0.906 m | 填補空間覆蓋缺口 |
| Heatmap + free-mask(×5) | 0.883 m | 分類取代回歸 |
| **Cascade ×5-ens** | **0.793 m** | 粗網格守門細網格 ── 冠軍 |

完整演進(含失敗路線)見 [lab3/EVOLUTION.md](lab3/EVOLUTION.md)。

```bash
# 重現冠軍數字(用已 commit 的權重)
cd lab3 && python3 load_best_model.py        # → median 0.793 m
# 實驗室即時 demo(ESP32 + RViz)
cd lab3 && ./run_at_lab.sh                    # 或 ./run_at_lab.sh replay
```

## 視覺化(Lab 2,github 直接看)

| 早上軌跡 | 晚上軌跡 |
|:---:|:---:|
| ![morning](lab2/visualizations/trajectories_overlay_morning.png) | ![evening](lab2/visualizations/trajectories_overlay_evening.png) |
| **全 221 條合併** | **RSSI combined-best** |
| ![combined](lab2/visualizations/trajectories_overlay_combined.png) | ![heatmap](lab2/visualizations/heatmap_combined_best.png) |
| **Dominant AP** | **早晚 RSSI 差異(BMELab +2.6 dBm)** |
| ![dominant](lab2/visualizations/heatmap_dominant_ap.png) | ![diff](lab2/visualizations/morning_vs_evening/diff_03_BMELab.png) |

## License

MIT,見 [LICENSE](LICENSE)。
