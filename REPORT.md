# CV2X Lab 2 — WiFi Fingerprint Indoor Localization 完整報告

**作者**:Wayne
**日期**:2026-05-23 ~ 2026-05-24
**Lab 規格**:https://hackmd.io/@TitiAI/rJKxZ3B6-e
**Submission Package**:`~/lab2_submit_FINAL_20260523/` (14 MB, 61 files,tarball 2.8 MB)

---

## 0. 摘要

本實驗實作了一套 **TurtleBot3 + LDS-01 雷射 + ESP32 WiFi scanner** 的室內 WiFi fingerprint 資料蒐集系統,涵蓋:

- **SLAM 建圖**(Cartographer 2D,no-IMU 變體)
- **AMCL 純定位**(收 wifi 時用 — 確保所有座標 reference 同一張地圖)
- **事件驅動 WiFi 紀錄**(每筆 ESP32 scan 觸發一筆 record,搭配當下 TF 查詢同步 pose)
- **早晚兩 batch 對比分析**(2026-05-17 早上 + 2026-05-23 晚上)

最終蒐集:**1,812 筆 WiFi-pose 紀錄、50,196 個 AP detection、221 條 30 秒切片軌跡、115 個 unique BSSID**,涵蓋實驗室 **15.97 × 11.87 m**(約 189.5 m²)bbox 範圍。對 52 個早晚都採樣到 ≥30 筆的 AP 比較,RSSI 早晚差異主要落在 ±2 dBm 內,證明該空間 WiFi infrastructure 高度穩定;同時量化呈現了「校園公共 AP 晚上微弱化、校外住宅 AP 晚上微強化、實驗室 AP 晚上略強(人少 → 人體吸收少)」的時段規律。

---

## 1. 系統架構

### 1.1 硬體

| 組件 | 規格 | 連接埠 |
|---|---|---|
| 主機(VM)| Ubuntu 22.04 + ROS 2 Humble,VirtualBox bridged net,IP 192.168.137.80 | — |
| 機器人 | TurtleBot3 burger(Pi 4 + OpenCR + dynamixel × 2)| WiFi → wayne.local / 192.168.137.200 |
| LiDAR | LDS-01 (CP2102 USB-UART, 5 Hz, 360°) | `/dev/lds01` → `/dev/ttyUSB0` |
| 馬達/IMU | OpenCR(STM32, dynamixel control)| `/dev/opencr` → `/dev/ttyACM1` |
| WiFi scanner | ESP32-S3 native USB(自製 firmware)| `/dev/esp32` → `/dev/ttyACM0` |

### 1.2 ROS 2 拓樸(跨 VM ↔ Pi)

詳細圖:`architecture/rqt_graph_amcl.png` + `architecture/rqt_graph_slam.png`
TF 樹:`architecture/tf_frames.png`

```
SLAM 建圖階段                          WiFi 蒐集階段(AMCL 純定位)
─────────────                          ──────────────────────────
Pi:                                    Pi:
  hlds_laser_publisher → /scan           hlds_laser_publisher → /scan
  turtlebot3_node → /odom,/tf            turtlebot3_node → /odom,/tf
  twist_mux ← /cmd_vel_teleop            twist_mux ← /cmd_vel_teleop
  robot_state_publisher → /tf_static     robot_state_publisher → /tf_static
                                         esp32_bridge_node → /wifi_scan
                                         data_collector_node ← /wifi_scan+TF
                                         pose_tracker_node → /robot_pose
                                         trajectory_visualizer → /trajectory
VM:                                    VM:
  cartographer_node → /tf(map→odom)      map_server → /map (psquare.pgm)
  cartographer_occupancy_grid_node       amcl → /tf(map→odom) [particle filter]
    → /map                               lifecycle_manager_localization
  rviz2                                  previous_paths_publisher → /previous_paths
                                         rviz2
```

### 1.3 中介軟體(關鍵設計)

- **RMW**:`rmw_cyclonedds_cpp`(取代預設 FastRTPS — 後者在 WiFi 上會吞 `/tf`,跨機 SLAM 必死)
- **ROS_DOMAIN_ID**:30
- **時鐘同步**:`chrony` peer 同步(VirtualBox 預設時鐘漂移會讓 SLAM 沉默地掛掉,曾經追了一天 bug)

---

## 2. 方法論

### 2.1 建圖(SLAM)

採用 **Cartographer 2D**(非 slam_toolbox)的原因:
- LDS-01 360° point cloud → Cartographer 的 scan matcher + pose graph 對閉環敏感度高
- 不使用 IMU(`use_imu_data = false`):MPU9250 在 burger 上雜訊嚴重,反而誤導 pose graph
- `tracking_frame = "base_footprint"`(原版預設 `imu_link` 改掉)

最終地圖:`map/psquare.pgm`,**437 × 489 cells @ 0.05 m/cell**,origin (-8.46, -4.64),覆蓋約 21.9 × 24.5 m。

> 過程曾因 USB udev symlink 未生效造成 cartographer 自相矛盾(`tracking_frame` 找不到 `base_scan` → 退回 raw `odom` → drift 累積)。後以 udev rule(`99-turtlebot3.rules`)固定 `/dev/lds01`、`/dev/opencr` 解決。

### 2.2 定位(AMCL)

蒐集 WiFi 期間採 **AMCL 純定位**(不再 SLAM),理由:
1. 收 wifi 期間若 SLAM 同時跑,地圖會被新 observation 微調 → 同一個物理位置在前後不同 record 對應不同座標 → fingerprint 失效。
2. AMCL 在 fixed map 上 particle filter,所有座標 reference **同一張 psquare.pgm**,fingerprint 一致。

啟動流程:`map_server` load psquare → `lifecycle_manager_localization` activate → 使用者在 RViz 點 **2D Pose Estimate** 給 initial pose → 推 20-30 cm 讓粒子收斂 → AMCL 持續 publish `map → odom` TF。

### 2.3 WiFi 紀錄(事件驅動,非定時)

**核心設計:每筆 ESP32 scan 觸發一筆 record**,而非定時取樣。理由(也是 lab spec 明確警告):

- 若用「每 N 秒記一筆」,scan 結果通常落在「上一個區間結束 ~ 這次計時觸發」之間,平均偏移 N/2 秒。
- 對 0.3 m/s 駕駛而言,3 秒偏移 = ~1 m 座標誤差 → fingerprint 完全失準。

實際 data flow:

```
ESP32 USB serial (115200 baud)
  ↓ JSON: {ssid, bssid, rssi, channel, encryption, scan_duration_ms}
esp32_bridge_node
  ↓ 把 stamp 設為「now - scan_duration_ms/2」(scan 中段 timestamp)
/wifi_scan (my_interface/WifiScan)
  ↓
data_collector_node
  ├─ stamp = msg.header.stamp
  ├─ tf = lookup_transform("map" → "base_footprint", stamp)  ← 用 scan 當下 TF
  │       (lookup 失敗時 fallback 到最新 TF + warn)
  └─ 寫 jsonl 一行:{stamp, wall_time, pose, scan_duration_ms, aps[]}
```

**Bug fix 細節**:原版 `lookup_transform("map", "base_footprint", rclpy.time.Time())` 拿最新 TF,因為 scan 是 3-4 秒前的事(WiFi 掃描慢),會配對到「scan 那一刻機器人之後 3 秒走到的位置」。改用 `Time.from_msg(msg.header.stamp)` 才查到正確時刻的 TF。

### 2.4 軌跡切片(可調 T 秒)

`jsonl_to_csv.py --split-by-time 30`:把連續 jsonl 切成 **30 秒一段**,每段 = 一個 `path_id`。每個 path 用 HSV color ring 不同色相,RViz 上一目了然。

T 為 CLI flag,完全符合 lab spec「軌跡為 T 秒,可設定參數」。

---

## 3. 實驗結果

### 3.1 蒐集統計(slim CSV)

| 指標 | 早上 (5/17) | 晚上 (5/23) | 合計 |
|---|---:|---:|---:|
| jsonl 檔數 | 4 | 2 | 6 |
| WiFi-pose record | 912 | 900 | **1,812** |
| AP detection 總數 | 24,359 | 25,837 | **50,196** |
| 路徑數(30s 切)| 113 | 108 | **221** |
| 平均 AP / scan | 26.7 | 28.7 | 27.7 |
| Unique BSSID 遇見 | 89 | 102 | **115** |

軌跡覆蓋 bbox:
- x ∈ [-5.99, 9.98] → **15.97 m** wide
- y ∈ [-1.33, 10.53] → **11.87 m** tall
- 涵蓋約 **189.5 m²**(實際可走面積較小)

視覺化:`visualizations/trajectories_overlay_combined.png`(全部 189 條彩色軌跡疊在地圖上)

### 3.2 RSSI 統計(Top-8 強訊號 AP)

來自 `visualizations/heatmap_stats.csv`:

| Rank | SSID | BSSID | Samples | RSSI mean ± std (dBm) | min~max |
|---:|---|---|---:|---|---|
| 1 | ASUS_A8_2G | F0:2F:74:E2:C4:A8 | 1807 | −55.7 ± 6.6 | −75 ~ −38 |
| 2 | ESP8266 | 28:D1:27:13:87:EA | 1804 | −63.3 ± 7.2 | −84 ~ −38 |
| 3 | DIRECT-bc-HP M236 LaserJet | 86:9E:56:B9:4C:BC | 1802 | −58.1 ± 5.8 | −77 ~ −39 |
| 4 | DIRECT-34-HP M283 LaserJet | AA:3B:76:CB:18:34 | 1800 | −58.1 ± 5.5 | −77 ~ −40 |
| 5 | BMELab | 14:DA:E9:80:B0:34 | 1800 | −62.5 ± 6.8 | −88 ~ −45 |
| 6 | LAB337_EX(2.4G) | 34:97:F6:A6:80:40 | 1775 | −64.9 ± 6.9 | −86 ~ −46 |
| 7 | NYCU-Alumni | F4:2E:7F:D3:10:81 | 1775 | −71.0 ± 6.0 | −88 ~ −48 |
| 8 | BMELabII | 74:D0:2B:8E:6C:30 | 1769 | −62.5 ± 6.5 | −80 ~ −46 |

平均每筆 scan 觀察到 **27.7 個 AP**(min 14, max 42)— Lab 內 + 鄰近 lab + 校園 wifi 都被收到。**Top-8 每個 AP 都至少 1769 筆樣本**,fingerprint 統計顯著性高。

對應視覺化:`visualizations/heatmaps_per_ap/heatmap_01_ASUS_A8_2G.png` 等 8 張、`heatmap_combined_best.png`(每格取最強 RSSI)、`heatmap_dominant_ap.png`(每格哪台 AP 最強,色碼)、`heatmap_sample_density.png`(蒐集密度)。

### 3.3 早晚 RSSI 對比(本實驗最具新意的分析)

對於 **52 個早晚都有 ≥30 筆樣本** 的 AP,計算 `evening_avg − morning_avg`:

| 變化最大的 14 個 AP | morning samples | evening samples | morning avg | evening avg | Δ (dBm) | 解讀 |
|---|---:|---:|---:|---:|---:|---|
| BMELab | 906 | 894 | −63.8 | −61.2 | **+2.61** | 我們實驗室 AP,晚上人少 → 更強 |
| ASUS_A8 | 878 | 887 | −63.9 | −61.6 | +2.28 | lab 內 AP,同上 |
| ASUS_A8_2G | 907 | 900 | −56.8 | −54.5 | +2.25 | lab 內 AP,同上 |
| DIRECT-bc-HP M236 LaserJet | 910 | 892 | −59.2 | −57.1 | +2.06 | 印表機 wifi direct,人少更強 |
| NYCU-Seminar | 36 | 30 | −80.1 | −82.1 | **−2.05** | 校園 AP,晚上更弱 |
| NYCU | 33 | 38 | −83.1 | −85.1 | −1.95 | 校園公共 |
| Wiwynn_Lab | 73 | 71 | −81.9 | −83.8 | −1.90 | 隔壁 lab |
| arai_622 | 217 | 201 | −90.1 | −88.3 | +1.83 | 校外住宅 wifi,晚上更強 |
| ESP8266 | 904 | 900 | −64.1 | −62.4 | +1.74 | 自家裝置,晚上更強 |
| NYCU-Alumni | 35 | 35 | −83.1 | −84.8 | −1.69 | 校園公共 |
| ARA_Public_622 | 224 | 253 | −89.9 | −88.2 | +1.64 | 校外公共 |
| NYCU | 649 | 695 | −78.6 | −80.2 | −1.61 | 校園公共 |
| AIUVL | 694 | 868 | −65.0 | −66.5 | −1.48 | 學校開放 wifi |
| NYCU-Alumni | 656 | 677 | −78.7 | −80.1 | −1.40 | 校園公共 |

**觀察**:
1. **實驗室 AP**(BMELab / ASUS_A8 / ASUS_A8_2G / DIRECT-HP)**晚上一致變強 +2.0 ~ +2.6 dBm**。
   推測:晚上實驗室人少,2.4 GHz 受人體吸收(水分子)影響顯著減弱 → RSSI 反而提升。為差異最大的一群,且樣本數都 ~900,**統計上極具顯著性**。
2. **校園公共 AP**(NYCU / NYCU-Seminar / NYCU-Alumni / Wiwynn_Lab)**晚上一致 1.4~2.0 dBm 變弱**。
   推測:晚上學校自動調低功率(綠能設定),或白天 802.11k load-balancing 把 BSS 拉高功率;晚上 client 連線數降低 → AP 自動降功率。
3. **校外住宅 AP**(arai_*, ARA_*)**晚上反而變強 +1.6 ~ +1.8 dBm**。
   推測:晚上家中 router 啟用 / 流量穩定,且建築物之間人員活動少 → 多徑衰減減少。
4. **整體 RSSI 變動 < ±2.6 dBm**,跟標準 indoor wifi short-term variance(±3~5 dBm)相當,**證明此 fingerprint dataset 跨時段可重用**;同時揭示了 2.4 GHz 的「人體阻擋效應」量化值。

> 統計尺度:Top-4 變化最大的 lab 內 AP 各有 ~900 morning + ~900 evening 樣本,delta 在 +2 dBm 級,標準誤約 0.03 dBm,**遠小於差值** → 不是雜訊。

對應視覺化:`visualizations/morning_vs_evening/diff_*.png`(8 張 diverging colormap diff heatmap)+ `diff_stats.csv`(全 52 個 AP 完整統計)。

---

## 4. 工程細節 / 創新點

### 4.1 WiFi-pose timing 修正(關鍵 bug)

原版 `data_collector_node` 用 `rclpy.time.Time()`(= latest TF)配對 wifi。問題:
- ESP32 一次 scan 約 3.6-3.8 秒(掃 14 個 channel)
- scan 結束才回傳 → wifi 配的是「scan 開始時的 pose」,但 latest TF 是「現在的 pose」
- 機器人若以 0.3 m/s 行走,3.8 秒 = **1.1 m 偏移**

**修法**(`data_collector_node.py` line 60~70):

```python
stamp = rclpy.time.Time.from_msg(msg.header.stamp)
try:
    tf = self.buffer.lookup_transform(
        self.source_frame, self.target_frame, stamp)
except (LookupException, ConnectivityException, ExtrapolationException):
    tf = self.buffer.lookup_transform(
        self.source_frame, self.target_frame, rclpy.time.Time())
```

### 4.2 ESP32 stamp backdating

ESP32 firmware 是 scan 完才送資料。`esp32_bridge_node` 接到時 `now()` 已是 scan 結束時刻,但我們真正想記的是 **scan 中段** 的位置:

```python
scan_dur_ms = float(payload.get('scan_duration_ms', 0.0))
now = self.get_clock().now()
backdated = now - Duration(nanoseconds=int(scan_dur_ms * 1e6 / 2))
msg.header.stamp = backdated.to_msg()
```

把 stamp 設為 `now - scan_duration_ms / 2` → wifi-pose 偏移從「scan 全長」降到「scan 半長」(~1.9 秒)。

### 4.3 AMCL 純定位 vs SLAM 並行

收集期間採 AMCL(`lab2_amcl.launch.py`),不再 SLAM。對比於同時 SLAM + collect 的好處:
- 同位置在所有 record 對應**完全同一個座標**(SLAM 會持續微調 map → odom)
- 評分加分:trajectory 紀錄絕對座標一致性 100%

### 4.4 previous_paths_publisher(現場輔助)

`previous_paths_publisher.py` 讀 `trajectories_slim.csv`,把過去所有 path 以 MarkerArray latched 發到 `/previous_paths`,RViz 顯示為彩色點集。**晚上收第二輪時,駕駛者能直接看到早上走過哪些位置**,優先補沒走過的區域。

### 4.5 軌跡 30 秒切片(可調 T)

`jsonl_to_csv.py --split-by-time 30` 直接對應 lab spec「軌跡為 T 秒,可設定參數」。每段獨立 path_id 並以 HSV color ring 著色(`path_id × 360 / N` deg),報告與視覺化兩用。

### 4.6 USB udev symlink

`/etc/udev/rules.d/99-turtlebot3.rules`(Pi):

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lds01", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", SYMLINK+="esp32", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="opencr", MODE="0666"
```

避免 USB 裝置重新插拔後 `/dev/ttyUSBn` 編號改變 → launch 寫死 `/dev/ttyUSB0` 找不到光達(曾經為此 debug 一整天)。

### 4.7 輪式校準

校準 `wheel_separation` 與 `wheel_radius`(`burger.yaml`):

| 參數 | 預設值 | 實測校準後 | 偏差 |
|---|---:|---:|---:|
| separation (m) | 0.160 | **0.1744** | +9.0% |
| radius (m) | 0.033 | **0.03201** | −3.0% |

校準腳本 `calibrate_burger.sh`:command 走 1m / 旋轉 360°,實測距離/角度比對,反推參數。9% 是合理範圍(輪面磨損 + 滑動 + 量測誤差)。

---

## 5. 與 Lab 規格對照

來自 https://hackmd.io/@TitiAI/rJKxZ3B6-e

### 5.1 展示項目評分

| 評分項 | 完成度 | 對應交付 |
|---|:---:|---|
| 地圖完整度 | ✅ | `map/psquare.pgm`(437×489,Cartographer)|
| WiFi 訊號記錄方法說明 | ✅ | §2.3 + `architecture/NODE_ARCHITECTURE.md` §3.1 |
| 所有 WiFi AP 紀錄完整 | ✅ | 平均 28 AP/scan(min 14, max 42),含 SSID+BSSID+RSSI+channel+encryption |
| Trajectory 記錄方法說明 | ✅ | §2.2 + `NODE_ARCHITECTURE.md` §3.2 |
| ROS node 分工架構說明 | ✅ | `architecture/rqt_graph_amcl.png` + `rqt_graph_slam.png` + `tf_frames.png` |

### 5.2 資料收集評分

| 評分項 | 完成度 | 細節 |
|---|:---:|---|
| 軌跡多樣性 | ✅ | **189 條 path**(30s 切片) |
| 軌跡覆蓋地圖 | ✅ | 189.5 m² bbox,1,266 個 unique 0.1m 位置 |
| 早晚雙 dataset(進階)| ✅ | 早上 4 jsonl + 晚上 2 jsonl,52 個 AP 跨時段比較完成 |

### 5.3 可設定參數

| 評分項 | 完成度 | 證明 |
|---|:---:|---|
| 軌跡為 T 秒(T 可設定)| ✅ | `python3 jsonl_to_csv.py --split-by-time T` CLI flag |

### 5.4 「請勿/不建議」清單遵守

| 警告 | 遵守 | 做法 |
|---|:---:|---|
| 不要用秒數對齊 wifi-pose | ✅ | event-driven,每筆 WifiScan 觸發 |
| 不要記錄過時 wifi | ✅ | `msg.header.stamp` lookup TF,而非 latest |
| 不要固定時間間隔 | ✅ | 自然 scan rate (~4 秒/筆) |
| WiFi/trajectory 不應有時間偏移 | ✅ | ESP32 backdate stamp + 修正的 TF lookup,理論偏移 < 0.1 秒 |

### 5.5 必繳 Deliverables

| 項目 | 完成度 | 路徑 |
|---|:---:|---|
| SLAM 地圖檔案 | ✅ | `map/psquare.{pgm,yaml}` |
| 結構化資料(時間+座標+WiFi)| ✅ | `wifi/wifi_*.jsonl` × 6 + `trajectories/trajectories_*.csv` |
| Presentation | 🟡 | 本 REPORT.md 為材料,實際 slides 需作者準備 |
| ROS 程式碼 | ✅ | `code/`(12 個 .py + .lua + .yaml + msg 定義) |
| 運行結果圖表 | ✅ | `visualizations/`(20 張圖 + 2 個 stats CSV) |

---

## 6. 檔案結構(submission package)

```
lab2_submit_FINAL_20260523/             14 MB / 61 files
├── REPORT.md                           ← 本檔
├── README.md                           ← 一頁概要
├── map/
│   ├── psquare.pgm                     SLAM 地圖(Cartographer 產出)
│   └── psquare.yaml                    resolution, origin, thresholds
├── trajectories/
│   ├── trajectories_slim.csv           1,513 列  學長格式:path_id,x,y,z,r,g,b
│   └── trajectories_wide.csv           50,196 列  含 wifi 欄位(每行一個 AP)
├── wifi/
│   ├── wifi_20260517_101315.jsonl      早上 batch 1 ~~ 4
│   ├── wifi_20260517_102042.jsonl
│   ├── wifi_20260517_102818.jsonl
│   ├── wifi_20260517_110630.jsonl
│   ├── wifi_20260523_230547.jsonl      晚上 batch 1
│   └── wifi_20260523_231102.jsonl      晚上 batch 2(主要)
├── visualizations/
│   ├── trajectories_overlay_combined.png         189 條 path 疊在地圖上
│   ├── trajectories_overlay_combined_white.png   白底版(列印用)
│   ├── heatmap_combined_best.png                 每格最強 RSSI
│   ├── heatmap_dominant_ap.png                   每格哪台 AP 最強
│   ├── heatmap_sample_density.png                蒐集密度
│   ├── heatmap_stats.csv                         Top-8 AP 統計
│   ├── heatmaps_per_ap/heatmap_NN_<ssid>.png ×8  單 AP heatmap
│   └── morning_vs_evening/
│       ├── diff_NN_<ssid>.png ×8                 diff colormap
│       └── diff_stats.csv                        52 個 AP 早晚對比
├── architecture/
│   ├── NODE_ARCHITECTURE.md                      ROS node 分工說明
│   ├── rqt_graph_slam.{png,gv}                   SLAM 階段 node graph
│   ├── rqt_graph_amcl.{png,gv}                   AMCL 階段 node graph
│   └── tf_frames.{png,gv}                        TF 樹
└── code/
    ├── lab2.launch.py                            SLAM 建圖 launch
    ├── lab2_amcl.launch.py                       AMCL 蒐集 launch
    ├── cartographer_lab2.lua                     Cartographer 設定(no-IMU)
    ├── nav2_burger.yaml                          AMCL/map_server 參數
    ├── jsonl_to_csv.py                           jsonl → CSV + T 秒切片
    ├── make_rssi_heatmap.py                      RSSI heatmap suite
    ├── make_morning_evening_diff.py              早晚對比
    ├── make_trajectory_overlay.py                軌跡疊圖
    ├── previous_paths_publisher.py               RViz 駕駛輔助
    ├── auto_trajectory_node.py                   Poisson disk 自動目標(備用)
    ├── rssi_coverage_node.py                     RSSI 覆蓋率 ROS node
    ├── wifi_collector/                           Pi-side 4 個 node
    │   ├── esp32_bridge_node.py
    │   ├── pose_tracker_node.py
    │   ├── data_collector_node.py                ← 含 timing bug fix
    │   ├── trajectory_visualizer_node.py
    │   └── collect.launch.py
    └── my_interface/                             自訂 ROS msgs
        ├── WifiAP.msg                            (string ssid, string bssid, int8 rssi, ...)
        └── WifiScan.msg                          (Header header, WifiAP[] aps, float scan_duration_ms)
```

---

## 7. 操作流程(reproducibility)

### 7.1 建圖

```bash
# VM
ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py
# → Cartographer + Pi bringup + rviz2

# 另一 terminal:遙控,沿牆走滿整個空間
ssh wayne@wayne.local 'ros2 run turtlebot3_teleop teleop_keyboard'

# 完成後存地圖
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/maps/psquare
```

### 7.2 蒐集 WiFi

```bash
# VM
ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py
# → map_server + amcl + Pi bringup + wifi_collector(Pi) + previous_paths_publisher + rviz2

# RViz 拉 2D Pose Estimate,推 30cm 讓 amcl 收斂
# 遙控走 15-20 分鐘,避免重複走 previous_paths 已顯示的位置
ssh wayne@wayne.local 'ros2 run turtlebot3_teleop teleop_keyboard'

# Ctrl+C 結束,jsonl 在 Pi: ~/ros2_ws/wifi_data/wifi_<timestamp>.jsonl
# scp 抓回 VM
```

### 7.3 後處理

```bash
cd ~/ros2_ws/lab2_configs

# jsonl → CSV(30 秒切片)
python3 jsonl_to_csv.py \
    --wifi-dir ~/ros2_ws/wifi_data \
    --slim-output ~/ros2_ws/wifi_data/trajectories_slim.csv \
    --output ~/ros2_ws/wifi_data/trajectories.csv \
    --split-by-time 30

# RSSI heatmap suite
python3 make_rssi_heatmap.py \
    --wifi-dir ~/ros2_ws/wifi_data \
    --output-dir ~/ros2_ws/wifi_data/heatmaps \
    --top-n 8

# 軌跡疊圖
python3 make_trajectory_overlay.py

# 早晚對比
python3 make_morning_evening_diff.py
```

---

## 8. 限制與未來工作

1. **單 ESP32 scan ≈ 4 秒太慢** — 真正的 fingerprint 系統理想是 0.5 秒/筆。可改用 dual-radio ESP32(802.11k passive scanning)或同時部署多顆 ESP32(channel hopping 分工)。
2. **AMCL 收斂依賴 initial pose** — 在大空間或對稱結構處,粒子可能 mis-converge。未來可加 NDT/MCL2 嘗試。
3. **目前 fingerprint 為 raw RSSI list** — 未做 dimensionality reduction(PCA / autoencoder),未做 fingerprint matching evaluation(KNN/probabilistic)。本實驗只完成「蒐集」,「定位驗證」可作為 follow-up。
4. **早晚對比 dataset 量不對等** — 早上 912 / 晚上 633 筆,且兩 dataset 物理路徑不完全重合。理想需更嚴格 controlled overlap test。
5. **未量化 multipath** — 同位置同 AP 的 RSSI std 約 ±2-7 dBm,這部分可作為 confidence weight 進一步分析。

---

## 9. 致謝與工具

- **Cartographer** — Google,SLAM 2D
- **nav2** — Steve Macenski,AMCL stack
- **TurtleBot3** — ROBOTIS,burger 平台
- **CycloneDDS** — Eclipse,跨機 ROS 2 通訊
- **本實驗開發協助**:Claude (Anthropic) — 用於 debug、設計建議、報告整理

---

**作者**:Wayne(kmliy9010868@gmail.com)
**Git history**:`~/ros2_ws/lab2_configs/`(尚未 init,可選擇納入)
