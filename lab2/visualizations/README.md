# `visualizations/` — Lab 2 資料視覺化

由 `../code/make_*.py` 從 [`../../wifi/`](../../wifi/) 與
[`../../trajectories/`](../../trajectories/) 產生,疊在
[`../../map/psquare.pgm`](../../map/) 上。

| 檔案 | 內容 |
|---|---|
| `trajectories_overlay_morning/evening/combined.png` | 軌跡疊圖(早 / 晚 / 合併 221 段),HSV 環每段一色 |
| `trajectories_overlay_combined_white.png` | 合併版白底(列印用) |
| `heatmap_combined_best.png` | 每格取最強 AP 的 RSSI 熱圖 |
| `heatmap_dominant_ap.png` | 每格最常出現的 AP |
| `heatmap_sample_density.png` | 取樣密度(哪裡走得多) |
| `heatmap_stats.csv` | 熱圖統計數值 |
| `heatmaps_per_ap/` | 各 AP 各一張 RSSI 熱圖 |
| `morning_vs_evening/` | 各 AP 早晚 RSSI 差異圖(看出 BMELab 晚上 +2.6 dBm) |

重新產生:見 [`../README.md`](../README.md) 的 Quick start。
