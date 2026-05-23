# Presentation 講稿 / Slide outline

報告當天可以照這個順序講。每張 slide 一個 ##。建議 12 張,每張 2 分鐘。

---

## 1. Title slide

**CV2X Lab 2 — WiFi Fingerprint Indoor Localization**
- 學生:Wayne
- 平台:TurtleBot3 burger + LDS-01 + ESP32-S3
- 日期:2026-05-23

---

## 2. Problem statement

- GPS 在室內訊號弱 → 需要替代 indoor localization 手段
- WiFi fingerprint:在已知位置記錄 AP RSSI,推論未知位置時與資料庫比對
- 本實驗目標:**蒐集** 一整套 fingerprint dataset(訓練資料)

---

## 3. System overview

放架構圖 `architecture/rqt_graph_amcl.png` 或 NODE_ARCHITECTURE.md 第一張 ASCII 圖。

- Pi:LDS-01(/scan)、OpenCR(/odom,/tf)、ESP32(/wifi_scan)
- VM:Cartographer(建圖)→ AMCL(蒐集時定位)
- 跨機通訊:**CycloneDDS over WiFi**(取代不可靠的 FastRTPS)

---

## 4. Map building (Cartographer)

放 `map/psquare.pgm` 大圖。

- Cartographer 2D scan matching + pose graph
- **不用 IMU**(MPU9250 雜訊重)
- 437×489 cells @ 0.05 m/cell ≈ 22 × 24 m 室內空間

---

## 5. WiFi 紀錄方法(評分重點)

放 NODE_ARCHITECTURE.md §3.1 dataflow ASCII 圖。

關鍵設計 — **event-driven**(非 timer-driven):
> 「有新的 WiFi 結果就紀錄座標與 WiFi」(spec 原文)

而非每 N 秒記一筆(spec 明確警告)。

---

## 6. Trajectory 紀錄方法(評分重點)

放 NODE_ARCHITECTURE.md §3.2 圖。

- Cartographer 建好的 map → AMCL particle filter 在 map 上定位
- `data_collector_node` 用 **scan 當下的 stamp** lookup TF(map → base_footprint)
- 每筆 wifi 配對到**正確時刻**的 pose,非 latest TF

---

## 7. Bug fix highlight(關鍵創新)

**Lab spec 明文警告的 bug,我們真的修了:**

原本 `data_collector_node` 用 `rclpy.time.Time()`(latest TF)配 wifi:
- ESP32 scan ~3.8 秒,wifi 紀錄是「scan 開始時」,latest TF 是「現在」
- 機器人若以 0.3 m/s 走,**位置偏移 1.1 m**

修法:`lookup_transform(t=msg.header.stamp)`,加上 fallback。

另外 ESP32 stamp 設為「now - scan_dur/2」(backdating)→ 偏移降到 < 0.1 s。

---

## 8. ROS node 分工(評分重點)

放 `architecture/rqt_graph_amcl.png` + `rqt_graph_slam.png`(可拆兩張)。

| 階段 | VM | Pi |
|---|---|---|
| 建圖 | Cartographer + RViz | bringup + twist_mux |
| 蒐集 | map_server + AMCL + RViz | bringup + 4×wifi_collector node |

明確分工:**運算密集在 VM,sensor + actuator 在 Pi**。

---

## 9. Result — 蒐集量

放 `visualizations/trajectories_overlay_combined.png`。

- **221 條軌跡**(早上 113 + 晚上 108,30 秒切片)
- **1,812 筆 WiFi-pose record**
- **50,196 個 AP detection**
- **115 個 unique BSSID**
- 涵蓋 15.97 × 11.87 m bbox = 189.5 m²

---

## 10. Result — RSSI 分佈

放 `visualizations/heatmap_combined_best.png` + `heatmap_dominant_ap.png`。

- Top-8 強訊號 AP:ASUS_A8_2G 最強(−55.7 dBm),樣本 1807 筆
- 印表機 WiFi Direct(DIRECT-HP)也很強(實驗室就有兩台)
- 公共校園 wifi(NYCU)較弱(−71 dBm)

---

## 11. 加分項 — 早晚對比

放 `visualizations/morning_vs_evening/diff_03_BMELab.png`。

對 52 個 AP(早晚都有 ≥30 筆樣本)算 `evening_avg - morning_avg`:

| 類型 | Δ 範圍 | 解讀 |
|---|---|---|
| 實驗室內 AP | **+2.0 ~ +2.6 dBm** | 晚上人少 → 2.4 GHz 人體吸收減少 |
| 校園公共 AP | −1.4 ~ −2.0 dBm | 學校自動降功率 |
| 校外住宅 AP | +1.6 ~ +1.8 dBm | 室內活動少 → 多徑衰減少 |

整體 < ±2.6 dBm → **dataset 跨時段可重用**。

---

## 12. Limitations + Future work

- 單 ESP32 scan ~4 秒,理想 fingerprint 應 0.5 秒/筆 → 改 dual-radio
- 本實驗只完成「蒐集」,「定位驗證」(KNN / DNN matching)可做 follow-up
- 早晚 dataset 路徑未嚴格 controlled overlap → 未來可用 stratified sampling

謝謝。

---

## 講者筆記

- 若被問「為什麼用 AMCL 不用 SLAM 同時跑」→ §2.2 答案:座標 reference 一致性
- 若被問「為什麼不用 IMU」→ MPU9250 noisy + cartographer use_imu_data 設 false
- 若被問「scan rate 為何 ~4 秒這麼慢」→ ESP32 802.11 active scan 全 14 個 channel,每 channel ~150 ms dwell
- 若被問「最大的 debug 工程」→ FastRTPS over WiFi 吞 /tf;改 CycloneDDS 解決
