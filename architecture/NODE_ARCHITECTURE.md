# CV2X Lab 2 — ROS Node 分工架構

## 一、整體架構概覽

```
┌─────────────────────── 🤖 Pi (wayne@wayne.local) ───────────────────────┐
│                                                                          │
│  HARDWARE:  LDS-01 lidar │ OpenCR + wheel encoders │ ESP32 wifi scanner  │
│                  │                  │                       │            │
│                  ↓                  ↓                       ↓            │
│         /hlds_laser_publisher   /turtlebot3_node      /esp32_bridge_node │
│                  │                  │                       │            │
│                /scan         /odom +/tf(odom→base)    /wifi_scan         │
│                  │                  │                       │            │
│                  │                  ↓                       ↓            │
│                  │             /twist_mux ← /cmd_vel_teleop or /cmd_vel  │
│                  │                  │                                    │
│                  │                  ↓ /cmd_vel → 馬達                    │
│                  │                                                       │
│                  │                              /data_collector_node     │
│                  │                                       ↓               │
│                  │                            ~/ros2_ws/wifi_data/       │
│                  │                                 wifi_YYYY.jsonl       │
│                  │                                                       │
│                  │                              /pose_tracker_node       │
│                  │                                  → /robot_pose        │
│                  │                                                       │
│                  │                       /trajectory_visualizer_node     │
│                  │                                  → /trajectory        │
└──────────────────┼──────────────────────────────── (CycloneDDS / WiFi) ──┘
                   │     ↑                              ↑           ↑
                   ↓     │                              │           │
┌─────────────────────── 🖥️ VM (192.168.137.80) ───────────────────────────┐
│                                                                          │
│  /map_server  /map                                                       │
│      ↓        ↓                                                          │
│  /amcl ───── /tf (map→odom)                                              │
│  (particle filter; uses /scan + /odom)                                   │
│                                                                          │
│  /lifecycle_manager_localization  — activates map_server + amcl          │
│                                                                          │
│  /previous_paths_publisher  → /previous_paths (MarkerArray, latched)     │
│                                                                          │
│  /rviz2 — 2D Pose Estimate (→ /initialpose) + 所有顯示                   │
│                                                                          │
│  (Teleop) /teleop_keyboard → /cmd_vel_teleop                             │
└──────────────────────────────────────────────────────────────────────────┘
```

## 二、Node 對應評分項

| Node | 在哪 | 評分項 | 提供什麼 |
|------|----|------|-------|
| `hlds_laser_publisher` | Pi | 地圖完整度 / 軌跡 | `/scan` LDS-01 點雲 5 Hz |
| `turtlebot3_node` (含 diff_drive_controller) | Pi | 軌跡完整度 | `/odom` 20 Hz + `/tf` odom→base_footprint |
| `robot_state_publisher` | Pi | TF chain | `/tf_static` base_footprint↔base_scan↔imu_link |
| `twist_mux` | Pi | 控制優先級 | teleop(p=100) 蓋過 nav(p=10),avoid 衝突 |
| `esp32_bridge_node` | Pi | **如何記錄 WiFi 訊號** | serial → `/wifi_scan` event-driven |
| `pose_tracker_node` | Pi | **如何記錄 trajectory** | TF→ `/robot_pose` 10 Hz |
| `data_collector_node` | Pi | **WiFi+pose 配對 / jsonl 紀錄** | `/wifi_scan` × TF(at scan stamp)→ jsonl 一行一筆 |
| `trajectory_visualizer_node` | Pi | **trajectory 視覺** | TF → `/trajectory` Path |
| `map_server` | VM | 地圖完整度 | publish psquare 為 `/map` static |
| `amcl` | VM | **如何記錄 trajectory(絕對座標)** | particle filter `/map`+`/scan`+`/odom` → `/tf` map→odom |
| `lifecycle_manager_localization` | VM | node 分工 | 統一管 map_server + amcl 生命週期 |
| `previous_paths_publisher` | VM | trajectory 視覺 / 駕駛輔助 | 讀 CSV → `/previous_paths` MarkerArray latched |
| `rviz2` | VM | 視覺化整合 | 顯示 map + scan + tf + trajectory + previous_paths + 2D Pose Estimate |

## 三、資料流(把以上連起來)

### 3.1 WiFi 收集流(對應「如何記錄 WiFi 訊號」)
```
ESP32 USB 序列 → esp32_bridge_node → /wifi_scan(my_interface/WifiScan)
                                          ↓
                                  data_collector_node
                                          ↓ (查 TF at scan stamp)
                                  pose+wifi 配對
                                          ↓
                          ~/wifi_data/wifi_TIMESTAMP.jsonl
```
**關鍵設計:event-driven**(每筆 WifiScan 觸發一筆紀錄),非 timer-driven —
完全符合 lab spec「即為有新的 WiFi 結果就紀錄座標與 WiFi」。

### 3.2 Trajectory 紀錄流(對應「如何記錄 trajectory」)
```
[輪式編碼器] turtlebot3_node → /tf (odom→base_footprint) 20Hz
[LDS-01]    hlds_laser → /scan 5Hz   ┐
[psquare 地圖] map_server → /map      ┴→ amcl → /tf (map→odom)
                                              ↓
                          pose_tracker_node 查 TF(map→base_footprint)
                                              ↓
                          jsonl 每筆有絕對座標 + yaw
```

### 3.3 可設定參數(對應「軌跡為 T 秒」)
```
jsonl_to_csv.py --split-by-time T_SECONDS
   把連續 jsonl 切成 T 秒一段,每段 = 新 path_id + HSV 顏色
   T 為 launch arg / CLI flag,完全可調
```

## 四、創新點(超出 lab 基本要求)

| 創新 | 對應評分項加分 |
|-----|--------------|
| **WiFi-pose timing fix** — 修了原版 data_collector 用「最新 TF」配「3 秒前 wifi」的 bug,改用 `msg.header.stamp` 查 TF | 如何記錄 WiFi 訊號 |
| **esp32_bridge backdated stamp** — 用 `now - scan_duration_ms/2` 把 stamp 設為 scan 中段,進一步減少 wifi-pose 時間偏移 | 同上 |
| **AMCL 純定位**(不再 SLAM)— 收 wifi 期間 map 完全不變,所有 wifi 點 reference 同一坐標系 | trajectory 完整度 |
| **previous_paths_publisher**(現場輔助)— 開車時 RViz 顯示「之前 113 條 path」,避免重複走 | trajectory 多樣性 / 覆蓋率 |
| **make_rssi_heatmap.py** — per-AP RSSI heatmap + sample density + dominant-AP map + stats CSV | 如何記錄 WiFi 訊號 |
| **多 T 切片可設定** — `--split-by-time 30` 或任何秒數,T 真的是 launch arg | 可設定參數 |

## 五、檢查指令(評分人想自己驗證的話)

```bash
# 啟動完整 stack
ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py

# 全部跑著的 node
ros2 node list

# 每個 node 細節
ros2 node info /amcl
ros2 node info /data_collector_node
...

# TF tree
ros2 run tf2_tools view_frames     # 產生 frames.pdf

# Node graph
rqt_graph                           # GUI 顯示 node↔topic
```
