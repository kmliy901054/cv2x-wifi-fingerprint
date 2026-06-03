# Lab 2 — WiFi 指紋資料蒐集

用 TurtleBot3 burger + LDS-01 + ESP32-S3,在 NYCU BME Lab(約 189.5 m²)
邊定位邊掃 WiFi,產出一份 (RSSI, pose) 指紋資料集。這是整個專案的「蒐集」階段;
下游的「使用」階段(深度學習定位)在 [`../lab3/`](../lab3/)。

## 文件導覽

| 文件 | 內容 |
|---|---|
| [REPORT.md](REPORT.md) | 完整報告(方法、設計決策、結果) |
| [SETUP.md](SETUP.md) | 環境建置與重現步驟 |
| [PRESENTATION.md](PRESENTATION.md) | 簡報講稿(搭配 `CV2X_Lab2_presentation.pptx`) |
| [architecture/NODE_ARCHITECTURE.md](architecture/NODE_ARCHITECTURE.md) | ROS 節點 / TF / rqt 圖 |

## 產出的資料集(在 repo 根目錄,與 Lab 3 共用)

| 路徑 | 內容 |
|---|---|
| [`../wifi/`](../wifi/) | 原始指紋 jsonl(6 檔、1,812 筆 scan、早 912 / 晚 900) |
| [`../trajectories/`](../trajectories/) | 表格化 CSV(slim 1,513 列 / wide 50,196 列) |
| [`../map/`](../map/) | SLAM 樓層平面圖 `psquare.pgm` |

蒐集數字:1,812 筆 record、50,196 次 AP 偵測、115 個 unique BSSID、221 條軌跡段。

## 程式(`code/`)

```
code/
  lab2.launch.py            SLAM 建圖(Cartographer)
  lab2_amcl.launch.py       AMCL 定位 + WiFi 蒐集
  cartographer_lab2.lua     Cartographer 調參(burger, IMU off)
  nav2_burger.yaml          nav2 / AMCL 參數
  calibrated_burger.yaml    輪徑校正
  wifi_collector/           ESP32 WiFi 蒐集 ROS 套件
  my_interface/             自訂 ROS message
  jsonl_to_csv.py           jsonl → trajectories CSV(可 --split-by-time T)
  make_rssi_heatmap.py      RSSI 熱圖
  make_morning_evening_diff.py / make_overall_diff.py   早晚差異圖
  make_trajectory_overlay.py / make_trajectory_split.py  軌跡疊圖
  previous_paths_publisher.py / auto_trajectory_node.py / rssi_coverage_node.py
```

## Quick start

```bash
# 建圖
ros2 launch code/lab2.launch.py
# 定位 + 蒐集
ros2 launch code/lab2_amcl.launch.py
# 後處理(產 CSV 與視覺化)
cd code
python3 jsonl_to_csv.py --split-by-time 30
python3 make_rssi_heatmap.py
python3 make_trajectory_overlay.py
```

詳細見 [SETUP.md](SETUP.md)。設計重點(摘要,完整在 REPORT.md):

- **兩階段拆開**:先 SLAM 建固定地圖,再 AMCL 純定位收 wifi —— 同一物理位置在所有
  record 對應同一座標,fingerprint 才有意義。
- **時間戳對齊**:用 `lookup_transform(..., msg.header.stamp)` 而非最新 TF;
  ESP32 一輪掃 ~3.8 s,並把 stamp 設成 `now - scan_dur/2` 對到掃描中段位置。
- **跨機用 CycloneDDS**:FastRTPS 在 WiFi 上 multicast 不穩,跨機 /tf 會掉。
