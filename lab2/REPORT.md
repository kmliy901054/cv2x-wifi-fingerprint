# CV2X Lab 2 — WiFi Fingerprint Indoor Localization

學生:Wayne
日期:2026 年 5 月
平台:TurtleBot3 burger + LDS-01 + ESP32-S3
ROS 2 Humble + Ubuntu 22.04

## 0. 摘要

本實驗在實驗室空間蒐集了一份 WiFi fingerprint 資料,作為室內定位的訓練資料。流程分兩階段:先用 SLAM 把空間建成一張固定的地圖,然後再用 AMCL 在這張地圖上做純定位,邊走邊讓 ESP32 掃 WiFi、邊把座標跟掃描結果配對寫進檔案。

蒐集兩次:早上(5/17)912 筆、晚上(5/23)900 筆,合計 **1,812 筆 record / 50,196 個 AP detection / 115 個 unique BSSID**。覆蓋 bbox 15.97 × 11.87 m(189.5 m²)。

對 52 個早晚都 ≥30 筆的 AP 做了 RSSI 對比分析,發現實驗室內 AP 晚上 RSSI 一致升高 2~2.6 dBm(人少時 2.4 GHz 受人體吸收少),校園公共 AP 晚上一致降 1.4~2 dBm(基地台 load-balancing 調功率)。整體變化 < ±2.6 dBm,跟 indoor wifi 的 short-term variance 同數量級。

## 1. 硬體

VM 端:Ubuntu 22.04 跑在 VirtualBox 上,bridged 網路 IP 192.168.137.80,跑 ROS 2 Humble 桌面版。
Pi 端:Raspberry Pi 4 + Ubuntu 22.04 server,跟 VM 同網段。

機器人組件:
- TurtleBot3 burger 平台
- OpenCR(STM32 馬達控制板)— 走 dynamixel 協議,接兩顆差速輪馬達
- LDS-01 360° 雷射雷達 — CP2102 USB-UART,5 Hz scan rate,~3.5 m 範圍
- ESP32-S3 開發板 — 自製 firmware,native USB CDC,每輪 scan 全 14 個 channel,~3.6-3.8 秒/筆

跨機通訊用 CycloneDDS(`rmw_cyclonedds_cpp`)。本來用 ROS 2 humble 預設的 FastRTPS,但在 WiFi 上 multicast 不可靠,跨機 `/tf` 會掉包,導致 SLAM 跟 AMCL 全部掛掉。換 CycloneDDS 之後穩定。

USB 裝置用 udev rule 固定 symlink,避免重新插拔後 `/dev/ttyUSB0` / `ttyACM*` 編號跳掉。具體規則:LDS-01 → `/dev/lds01`、OpenCR → `/dev/opencr`、ESP32 → `/dev/esp32`(以 USB vendor ID 識別,不靠插入順序)。

時鐘同步用 `chrony`,VM 跟 Pi peer 對時。VirtualBox 預設時鐘會慢慢漂,沒同步的話 cartographer 的 scan 跟 odom timestamp 對不上,SLAM 會沉默地掛掉(沒 error,但 pose graph 不收斂)。

## 2. SLAM 建圖

### 2.1 工具選擇

用 Google Cartographer 2D,沒用 slam_toolbox。理由是 cartographer 對 360° 點雲的 scan matching 加上 pose graph 閉環優化敏感度比較高;slam_toolbox 在大空間 + LDS-01 這種低密度雷達上容易漂。Cartographer 的另一個好處是它把 `map → odom` TF 自己 publish,不用另外接 odom_tf_relay。

### 2.2 設定檔調整

預設 turtlebot3_cartographer 的 lua 是給 burger + IMU 的設定,但本實驗 IMU 關掉了。`code/cartographer_lab2.lua` 主要動兩處:

```lua
tracking_frame = "base_footprint"      -- 預設 imu_link,IMU 關了所以改 base_footprint
TRAJECTORY_BUILDER_2D.use_imu_data = false
```

不關 IMU 的話 cartographer 會卡在等 `/imu` topic 卻收不到(我們把 `/imu` remap 掉),或更糟,讀到雜訊大的 MPU9250 反而干擾 pose graph。MPU9250 在這顆 burger 上 yaw drift 大概 5°/min,SLAM 用就是災難。

### 2.3 操作流程

VM 端跑 `ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py`,內含:
1. SSH 上 Pi 開 bringup(robot_state_publisher + hlds_laser_publisher + turtlebot3_node + twist_mux)
2. VM 本地開 cartographer_node + cartographer_occupancy_grid_node
3. 開 RViz 顯示即時建圖過程

然後另開一個 terminal 跑 `teleop_keyboard` 遙控車子。掃描策略是沿牆繞,讓 lidar 看到夠多直線特徵供 scan matching 對齊。中途繞回起點至少一次,觸發閉環優化(cartographer 看到「我之前來過這裡」,把累積誤差攤平)。

建圖過程踩到的坑(重要,評分時可講):
- LDS-01 5V 線鬆過一次,馬達沒轉但 USB 還在,hlds_laser_publisher 卡在 `read()` 沒報錯
- udev symlink 過期(沒重新 trigger),導致 cartographer 找不到 lidar
- VM/Pi 時鐘漂太多,scan 跟 odom 對不上時間,pose graph 不收斂
這三個都遇過,後來加 SETUP.md 寫進去。

### 2.4 存圖

掃完後 Ctrl+C 結束 launch,再跑:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/maps/psquare
```

產出兩個檔:
- `psquare.pgm`:437 × 489 像素的灰階圖,黑 = 障礙,白 = 可走,灰 = 未觀測
- `psquare.yaml`:metadata,記了 resolution = 0.05 m/cell、origin = (-8.46, -4.64, 0)、占用閾值等

整張地圖實際物理範圍約 21.9 × 24.5 m。origin 對應到實驗室某個牆角(座標系定義時的零點)。

## 3. AMCL 純定位 + WiFi 蒐集

### 3.1 為什麼不 SLAM 邊建邊收?

第一版我就是這樣做的,結果發現一個問題:cartographer 持續微調 `map` frame,導致同一個物理位置在前後不同 record 中對應不同的座標。舉例:在房間中央掃了 wifi → 寫進 jsonl 是 (3.1, 0.2);過 30 秒回到同一點 → cartographer 把座標系修正了 → 變成 (2.9, 0.4)。fingerprint 的根基(相同位置應該有相似 RSSI)直接崩潰。

解法是改用兩階段:先固定一張地圖,然後用 AMCL 在這張地圖上做粒子濾波純定位。AMCL 不會動地圖,所有 wifi record 都 reference 同一個座標系。

### 3.2 AMCL 工作原理

AMCL = Adaptive Monte Carlo Localization。簡單講:
1. 把 N 個粒子(N ∈ [min_particles, max_particles])撒在 initial pose 周圍
2. 機器人移動時,每個粒子按 odometry 預測新位置(會加 motion noise)
3. 拿來新一筆 `/scan`,模擬「假設我在這個粒子位置,雷射應該看到什麼」,跟實際比對,算 likelihood
4. likelihood 高的粒子保留,低的淘汰並重新採樣
5. 收斂後 → 用 weighted average 算出機器人 pose,publish `map → odom` TF

`code/nav2_burger.yaml` 裡的 amcl 區塊有重要參數:
- `min_particles: 500` / `max_particles: 2000`
- `alpha1-5`:motion model 噪聲(平移/旋轉互相影響的程度)
- `laser_model_type: "likelihood_field"`
- `update_min_d: 0.20 m` / `update_min_a: 0.20 rad`:走多少距離 / 轉多少角度才更新一次粒子

### 3.3 啟動 launch

`ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py` 一條指令做這些事:

1. SSH 上 Pi 開 bringup(同 SLAM 階段)
2. VM 本地開 `map_server`(載入 psquare.yaml)
3. 開 `amcl` node
4. 開 `lifecycle_manager_localization`:把 map_server 跟 amcl 從 unconfigured 帶到 active 狀態
5. 20 秒後 SSH 上 Pi 開 wifi_collector(4 個 node)
6. VM 本地開 `previous_paths_publisher`(把過去走過的軌跡發到 RViz 當駕駛輔助)
7. 開 RViz

20 秒延遲是給 AMCL 跟 map_server 暖機,確保 wifi_collector 啟動時 TF chain 已經能 lookup map → base_footprint。

### 3.4 操作步驟

啟動 launch 之後,RViz 上會看到地圖載進來,但機器人的 pose 還沒對:粒子雲散在 initial pose 周圍幾公尺。手動步驟:

1. **2D Pose Estimate**(RViz 工具列)→ 在地圖上點機器人實際位置 + 拖出方向箭頭
2. AMCL 把粒子集中到那個 pose 附近
3. 用 teleop 推機器人走 30 cm 左右,讓 AMCL 根據新進來的 scan 跟 odom 收斂
4. 看 RViz 上的粒子雲縮成一小團 → 收斂完成,可以開始正式收 wifi

收斂後 AMCL 持續發 `map → odom` TF(5-10 Hz),所有 wifi record 拿到的就是這條鏈出來的絕對座標。

### 3.5 蒐集策略

第二輪(晚上)我有意重疊早上的部分路徑,目的是讓同一個物理位置兩個時段都有樣本,做早晚對比有意義。`previous_paths_publisher` 在這裡有用:它讀 `trajectories_slim.csv`(早上的 113 條 path)做成 MarkerArray latched 發出來,RViz 顯示為彩色點集 — 駕駛者直接看到「早上走過哪」,儘量沿著走。

每個 jsonl 自然分段:每按一次 Ctrl+C 結束 wifi_collector,下次重啟就開新檔。早上 4 個 session(中間檢查、補充等),晚上 2 個 session。每筆 record 一行 JSON,內容包含 stamp、wall_time、pose(frame_id="map", x, y, z, yaw, quaternion)、scan_duration_ms、aps 陣列(每個 AP 含 ssid / bssid / rssi / channel / encryption)。

## 4. WiFi 紀錄細節(評分重點)

### 4.1 為什麼要事件驅動

Lab spec 明確警告「不建議設定秒數來對齊 WiFi 和 trajectory」。原因是 ESP32 active scan 一輪全 14 個 channel 需要 3.6-3.8 秒(每個 channel 約 250 ms dwell time)。如果用「每 N 秒記一筆」的方式:
- N < 3.8:會跳過好幾個 scan,或在同一個 scan 結果上多次取樣(雜訊)
- N > 3.8:會錯過 scan,但即使取樣到也是 stale data

正確做法是「ESP32 每完成一輪 scan → 觸發一筆 record」。實作:`esp32_bridge_node` 從 USB serial 一行一行讀,每讀到一筆 JSON 就 publish `/wifi_scan` topic;`data_collector_node` 訂閱這個 topic,callback 裡完成 wifi-pose 配對 + 寫 jsonl。

完全 event-driven,沒有 timer。實際 record rate 約 0.26 Hz(每筆 ~3.8 秒)。

### 4.2 stamp 對齊(關鍵 bug 修正)

第一版的 `data_collector_node` 用這樣的程式:

```python
tf = self.buffer.lookup_transform(
    self.source_frame, self.target_frame, rclpy.time.Time())
```

`rclpy.time.Time()` 是「拿最新的 TF」。問題:wifi callback 觸發時,header.stamp 是「scan 結束那一刻」,但機器人已經繼續走了。如果我們把這個 wifi 配給「當下」的 TF,等於配給「scan 結束之後又走了一段路」的 pose。

對 0.3 m/s 駕駛 + 3.8 秒 scan 而言,空間偏移 ~1.1 m。fingerprint 完全錯位。Spec 提到「避免記錄過時的 WiFi」就是在講這個。

修法:

```python
stamp = rclpy.time.Time.from_msg(msg.header.stamp)
try:
    tf = self.buffer.lookup_transform(
        self.source_frame, self.target_frame, stamp)
except (LookupException, ConnectivityException, ExtrapolationException):
    # TF buffer 可能還沒有那麼舊的資料,fallback 最新
    tf = self.buffer.lookup_transform(
        self.source_frame, self.target_frame, rclpy.time.Time())
```

加上 fallback 是因為 ROS 2 的 TF buffer 預設只保留最近幾秒,scan stamp 太舊時會 throw `ExtrapolationException`。fallback 不理想但比 drop 整筆 record 好。

### 4.3 ESP32 stamp backdating

ESP32 firmware 是 scan 完才把整批結果 dump 到 serial。`esp32_bridge_node` 接到時 `now()` 已經是「scan 結束那一刻」。但我們要的其實是「scan 中段那一刻」的位置 — 因為 scan 期間機器人在動,單一 stamp 沒辦法精確表達整筆 scan 的「位置」,折衷取中間。

```python
scan_dur_ms = float(payload.get('scan_duration_ms', 0.0))
now = self.get_clock().now()
backdated = now - Duration(nanoseconds=int(scan_dur_ms * 1e6 / 2))
msg.header.stamp = backdated.to_msg()
```

把 stamp 設成 `now - scan_duration_ms / 2`。等於告訴下游「這筆 scan 對應的是『中間那一刻』的 pose」。配合 4.2 的 lookup,wifi-pose 時間偏移從「scan 全長」(~3.8 秒)降到「scan 半長」(~1.9 秒);再扣掉 lookup 那一瞬機器人移動的位置(< 0.05 m at 0.3 m/s),最終空間偏移 < 0.6 m,且是對「中間時刻」的最佳估計。

### 4.4 jsonl → CSV(T 秒切片)

`jsonl_to_csv.py` 把連續 jsonl 依時戳切成 T 秒一段,每段一個 path_id,色碼用 HSV ring(`h = pid × 137.508° mod 360`)讓相鄰 path 顯著不同色。

CLI flag `--split-by-time T`:T 完全可調,符合 spec「軌跡為 T 秒,可設定參數」。本繳交版用 30 秒。

兩種輸出:
- slim CSV(學長要的格式):`path_id, x, y, z, r, g, b`,1,513 列
- wide CSV(完整,給後續分析用):加 `yaw, ssid, bssid, rssi, channel, encryption, stamp_sec, stamp_nanosec`,50,196 列(每筆 record × 每個 AP 一行)

## 5. 結果

### 5.1 蒐集量

| 指標 | 早上 (5/17) | 晚上 (5/23) | 合計 |
|---|---:|---:|---:|
| jsonl 檔數 | 4 | 2 | 6 |
| WiFi-pose record | 912 | 900 | 1,812 |
| AP detection 總數 | 24,359 | 25,837 | 50,196 |
| 路徑數(30s 切)| 113 | 108 | 221 |
| 平均 AP / scan | 26.7 | 28.7 | 27.7 |
| Unique BSSID | 89 | 102 | 115 |

軌跡 bbox:x ∈ [-5.99, 9.98] = 15.97 m,y ∈ [-1.33, 10.53] = 11.87 m,bbox 面積 189.5 m²。實際可走面積較小(扣掉障礙物)。

對應視覺化:
- `visualizations/trajectories_overlay_morning.png` — 早上 112 條 path
- `visualizations/trajectories_overlay_evening.png` — 晚上 108 條 path
- `visualizations/trajectories_overlay_combined.png` — 全部 221 條

### 5.2 RSSI 統計(Top-8 強訊號)

來自 `visualizations/heatmap_stats.csv`:

| Rank | SSID | BSSID | Samples | mean ± std (dBm) | min~max |
|---:|---|---|---:|---|---|
| 1 | ASUS_A8_2G | F0:2F:74:E2:C4:A8 | 1807 | −55.7 ± 6.6 | −75 ~ −38 |
| 2 | ESP8266 | 28:D1:27:13:87:EA | 1804 | −63.3 ± 7.2 | −84 ~ −38 |
| 3 | DIRECT-bc-HP M236 LaserJet | 86:9E:56:B9:4C:BC | 1802 | −58.1 ± 5.8 | −77 ~ −39 |
| 4 | DIRECT-34-HP M283 LaserJet | AA:3B:76:CB:18:34 | 1800 | −58.1 ± 5.5 | −77 ~ −40 |
| 5 | BMELab | 14:DA:E9:80:B0:34 | 1800 | −62.5 ± 6.8 | −88 ~ −45 |
| 6 | LAB337_EX(2.4G) | 34:97:F6:A6:80:40 | 1775 | −64.9 ± 6.9 | −86 ~ −46 |
| 7 | NYCU-Alumni | F4:2E:7F:D3:10:81 | 1775 | −71.0 ± 6.0 | −88 ~ −48 |
| 8 | BMELabII | 74:D0:2B:8E:6C:30 | 1769 | −62.5 ± 6.5 | −80 ~ −46 |

Top-8 每個 AP 至少 1769 筆樣本,fingerprint 統計顯著。

對應視覺化:
- `visualizations/heatmaps_per_ap/heatmap_01..08_*.png` — 8 張單 AP RSSI 熱圖
- `visualizations/heatmap_combined_best.png` — 每格取最強 RSSI
- `visualizations/heatmap_dominant_ap.png` — 每格哪台 AP 最強(色碼)
- `visualizations/heatmap_sample_density.png` — 每格 scan 觸發次數

### 5.3 早晚 RSSI 對比

對 52 個早晚都 ≥30 筆樣本的 AP 算 `delta = evening_avg − morning_avg`:

| SSID | 早 # | 晚 # | 早 avg | 晚 avg | Δ (dBm) | 解讀 |
|---|---:|---:|---:|---:|---:|---|
| BMELab | 906 | 894 | −63.84 | −61.22 | +2.61 | 實驗室 AP,晚上人少 |
| ASUS_A8 | 878 | 887 | −63.92 | −61.64 | +2.28 | 實驗室 AP |
| ASUS_A8_2G | 907 | 900 | −56.79 | −54.54 | +2.25 | 實驗室 AP |
| DIRECT-bc-HP M236 LJ | 910 | 892 | −59.15 | −57.09 | +2.06 | 印表機 WiFi Direct |
| NYCU-Seminar | 36 | 30 | −80.08 | −82.13 | −2.05 | 校園 AP,晚上更弱 |
| NYCU | 33 | 38 | −83.15 | −85.11 | −1.95 | 校園公共 |
| Wiwynn_Lab | 73 | 71 | −81.88 | −83.77 | −1.90 | 隔壁 lab |
| arai_622 | 217 | 201 | −90.12 | −88.30 | +1.83 | 校外住宅 wifi |
| ESP8266 | 904 | 900 | −64.13 | −62.39 | +1.74 | 自家裝置 |
| NYCU-Alumni | 35 | 35 | −83.11 | −84.80 | −1.69 | 校園公共 |

物理解讀分三類:

實驗室內 AP(BMELab, ASUS_A8, ASUS_A8_2G, DIRECT-HP):晚上一致升 +2.0 ~ +2.6 dBm。實驗室晚上人少,2.4 GHz 受人體(水分子吸收 + 多徑散射)影響顯著減弱 → 同一台 AP 同一個位置,接收到的訊號變強。這四個 AP 各有 ~900 morning + ~900 evening 樣本,標準誤約 0.03 dBm,delta 在 +2 dBm 級遠超雜訊範圍。

校園公共 AP(NYCU, NYCU-Seminar, NYCU-Alumni, Wiwynn_Lab):晚上一致降 1.4 ~ 2.0 dBm。推測學校 enterprise AP 有自動功率調整,白天客戶端多 → 拉高功率 + load-balancing 把 RSSI 推高;晚上客戶端少 → 降功率。

校外住宅 AP(arai_*, ARA_*):晚上升 +1.6 ~ +1.8 dBm。推測晚上住宅樓內活動少,跨建築物的牆面 + 玻璃多徑衰減減少。

整體變動 < ±2.6 dBm,跟 indoor wifi 自然 short-term variance(±3-5 dBm)同數量級。結論:此 fingerprint dataset 跨時段可重用,但若做精確定位,實驗室 AP 應加入「人數」這個 latent variable 校正。

對應視覺化:`visualizations/morning_vs_evening/diff_*.png`(8 張 diverging colormap)+ `diff_stats.csv`(52 個 AP 完整對比表)。

## 6. ROS node 分工(評分項對應)

| Node | 在哪 | 提供什麼 | 對應評分項 |
|---|:---:|---|---|
| hlds_laser_publisher | Pi | `/scan` 5 Hz | 地圖完整度 / 軌跡 |
| turtlebot3_node | Pi | `/odom` 20 Hz + `/tf` odom→base_footprint | 軌跡 |
| robot_state_publisher | Pi | `/tf_static` chain | TF chain |
| twist_mux | Pi | 控制 priority routing | 控制安全 |
| esp32_bridge_node | Pi | `/wifi_scan` (event-driven) | 如何記錄 WiFi |
| pose_tracker_node | Pi | TF → `/robot_pose` 10 Hz | 如何記錄 trajectory |
| data_collector_node | Pi | wifi × TF(at scan stamp) → jsonl | WiFi+pose 配對 |
| trajectory_visualizer_node | Pi | TF → `/trajectory` Path | trajectory 視覺 |
| map_server | VM | psquare → `/map` | 地圖完整度 |
| amcl | VM | particle filter → `/tf` map→odom | 絕對座標 |
| lifecycle_manager_localization | VM | 統一管 map_server + amcl 啟停 | node 分工 |
| previous_paths_publisher | VM | CSV → `/previous_paths` MarkerArray | 駕駛輔助 |
| rviz2 | VM | 整合視覺化 | 視覺 |

完整的 rqt_graph 在 `architecture/rqt_graph_slam.png`(SLAM 階段)跟 `architecture/rqt_graph_amcl.png`(AMCL 蒐集階段)。TF 樹在 `architecture/tf_frames.png`。文字版說明在 `architecture/NODE_ARCHITECTURE.md`。

## 7. 與 Lab 規格對照

| Spec 要求 | 完成狀況 |
|---|:---:|
| TurtleBot3 SLAM 掃描空間地圖 | ✅ Cartographer 2D, psquare.pgm |
| 整合 ESP32 與 TurtleBot | ✅ esp32_bridge_node + /wifi_scan |
| 驅動軌跡並收集 WiFi | ✅ 221 條 path,event-driven |
| 軌跡與 RSSI 對應 | ✅ jsonl + wide CSV(配對絕對座標)|
| WiFi 必含 SSID + MAC + RSSI | ✅ 加贈 channel + encryption + scan_duration |
| 紀錄全部 WiFi | ✅ 平均 27.7 AP / scan,沒過濾 |
| Trajectory 必含時間 + 絕對座標 + WiFi | ✅ stamp + frame_id="map" + aps[] |
| 事件驅動(非秒數對齊)| ✅ subscribe `/wifi_scan` callback |
| 避免記錄過時 WiFi | ✅ TF lookup at scan stamp + ESP32 backdate |
| T 秒可配置 | ✅ `--split-by-time T` CLI flag |
| keyboard 控制 | ✅ teleop_keyboard + twist_mux |
| ROS 2 Humble + burger + LDS-01 | ✅ |
| ROS node 分工說明 | ✅ NODE_ARCHITECTURE.md + 兩張 rqt_graph |
| 軌跡多樣性 | ✅ 221 條 path,30s 切片 + HSV |
| 覆蓋地圖程度 | ✅ 189.5 m² bbox |
| 視覺化 RSSI 分佈 | ✅ 13 張 heatmap + 8 張早晚 diff |

## 8. 操作流程

### 8.1 建圖

```bash
ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py
```

VM 跑 Cartographer + Pi bringup。遙控走滿空間,沿牆 + 至少一次閉環。

存圖:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/maps/psquare
```

### 8.2 蒐集 WiFi

```bash
ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py
```

VM 跑 map_server + amcl + RViz,Pi 跑 bringup + wifi_collector(20 秒延遲)。RViz 拉 2D Pose Estimate,推 30 cm 收斂,然後遙控走。

### 8.3 後處理

```bash
cd ~/ros2_ws/lab2_configs
python3 jsonl_to_csv.py --split-by-time 30
python3 make_rssi_heatmap.py
python3 make_trajectory_overlay.py
python3 make_trajectory_split.py             # 早 / 晚分開的 overlay
python3 make_morning_evening_diff.py
```

## 9. 限制與改進

蒐集端:
- 單顆 ESP32 一輪 scan ~3.8 秒,蒐集密度受限。若要做正式 fingerprint matching,理想至少 1 Hz。可用 dual-radio ESP32 或多顆 ESP32 channel hopping 分工。
- AMCL 收斂依賴 initial pose。對稱結構或大空間有 mis-converge 風險。可加 NDT 或 MCL2 嘗試。

分析端:
- 本實驗只完成「蒐集」階段,實際定位(KNN / probabilistic / DNN matching)未做,可作為 follow-up。
- 早晚兩個 dataset 路徑不嚴格 controlled overlap。要嚴謹的時段對比需用 stratified sampling 規劃。

理論上:
- 早晚 RSSI 差異揭示「人體吸收」這個 latent variable。實驗室 AP 在「人數」這個變數上敏感,要做準確定位需要校正。
- 沒做 RSSI std-weighted matching。雜訊大的 AP(std > 7 dBm)應該降權。

## 10. 檔案結構

```
lab2_submit_FINAL_20260523/
├── REPORT.md                  本檔
├── README.md                  GitHub 首頁
├── SETUP.md                   從零重現指南
├── PRESENTATION.md            報告講稿
├── CV2X_Lab2_presentation.pptx 27 張投影片
├── LICENSE                    MIT
├── .gitignore
├── map/
│   ├── psquare.pgm            SLAM 地圖
│   ├── psquare.yaml           metadata
│   └── psquare.png            PNG 版本(給 pptx 用)
├── trajectories/
│   ├── trajectories_slim.csv  1,513 列 學長格式
│   └── trajectories_wide.csv  50,196 列 含 wifi 欄位
├── wifi/
│   └── wifi_*.jsonl × 6       原始紀錄
├── visualizations/
│   ├── trajectories_overlay_morning.png    早上 112 paths
│   ├── trajectories_overlay_evening.png    晚上 108 paths
│   ├── trajectories_overlay_combined.png   合併 221 paths
│   ├── trajectories_overlay_combined_white.png  白底版
│   ├── heatmap_combined_best.png
│   ├── heatmap_dominant_ap.png
│   ├── heatmap_sample_density.png
│   ├── heatmap_stats.csv
│   ├── heatmaps_per_ap/heatmap_NN_*.png × 8
│   └── morning_vs_evening/
│       ├── diff_NN_*.png × 8
│       └── diff_stats.csv
├── architecture/
│   ├── NODE_ARCHITECTURE.md
│   ├── rqt_graph_slam.png/.gv
│   ├── rqt_graph_amcl.png/.gv
│   └── tf_frames.png/.gv
└── code/
    ├── lab2.launch.py
    ├── lab2_amcl.launch.py
    ├── cartographer_lab2.lua
    ├── nav2_burger.yaml
    ├── calibrated_burger.yaml
    ├── jsonl_to_csv.py
    ├── make_rssi_heatmap.py
    ├── make_morning_evening_diff.py
    ├── make_trajectory_overlay.py
    ├── make_trajectory_split.py
    ├── make_pptx.py
    ├── previous_paths_publisher.py
    ├── auto_trajectory_node.py
    ├── rssi_coverage_node.py
    ├── wifi_collector/
    │   ├── esp32_bridge_node.py        含 stamp backdate
    │   ├── pose_tracker_node.py
    │   ├── data_collector_node.py      含 TF lookup at scan stamp
    │   ├── trajectory_visualizer_node.py
    │   ├── collect.launch.py
    │   └── __init__.py
    └── my_interface/
        ├── WifiAP.msg
        └── WifiScan.msg
```
