# 系統建置 / 重現指南

從零開始重現本實驗(VM + Pi + ESP32)需要的所有步驟。

---

## 環境

| 角色 | 規格 |
|---|---|
| **VM** | Ubuntu 22.04 LTS,VirtualBox bridged 網路,ROS 2 Humble |
| **Pi** | Raspberry Pi 4 + Ubuntu 22.04 + ROS 2 Humble |
| **TurtleBot3** | burger 平台 + OpenCR + LDS-01 |
| **WiFi scanner** | ESP32-S3 dev board(native USB)+ 自製 firmware |

---

## 1. VM 端設定

```bash
# ROS 2 Humble
sudo apt install ros-humble-desktop ros-humble-cartographer \
    ros-humble-cartographer-ros ros-humble-nav2-bringup \
    ros-humble-turtlebot3* ros-humble-rmw-cyclonedds-cpp

# 環境變數 ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
echo 'export LDS_MODEL=LDS-01' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=30' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc

# Workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <this repo>          # 把 code/ 內容對應放置(見下節)
colcon build
source install/setup.bash

# chrony(時鐘同步,跟 Pi 對時 — VirtualBox 預設時鐘漂移會讓 SLAM 沉默掛掉)
sudo apt install chrony
# 編輯 /etc/chrony/chrony.conf 加 'peer wayne.local'
sudo systemctl restart chrony
```

## 2. Pi 端設定

```bash
# ROS 2 Humble + TurtleBot3
sudo apt install ros-humble-ros-base ros-humble-turtlebot3-bringup \
    ros-humble-hls-lfcd-lds-driver ros-humble-rmw-cyclonedds-cpp \
    ros-humble-twist-mux

# 環境變數 ~/.bashrc(同 VM)
# ...略

# udev rules — 固定 USB 裝置名稱
sudo cp 99-turtlebot3.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
# 確認 /dev/lds01 /dev/opencr /dev/esp32 都存在
ls -la /dev/lds01 /dev/opencr /dev/esp32

# 校準 burger.yaml(用本 repo 的 calibrated_burger.yaml)
cp code/calibrated_burger.yaml \
   ~/turtlebot3_ws/src/turtlebot3/turtlebot3_bringup/param/humble/burger.yaml
cd ~/turtlebot3_ws && colcon build --packages-select turtlebot3_bringup

# wifi_collector package
mkdir -p ~/ros2_ws/src/wifi_collector
cp -r code/wifi_collector/* ~/ros2_ws/src/wifi_collector/
cp -r code/my_interface ~/ros2_ws/src/
cd ~/ros2_ws && colcon build
```

### udev rules `/etc/udev/rules.d/99-turtlebot3.rules`

```
# OpenCR (STMicro CDC)
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", \
    SYMLINK+="opencr", MODE="0666"
# ESP32-S3 native USB
SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", \
    SYMLINK+="esp32", MODE="0666"
# LDS-01 (CP2102)
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
    SYMLINK+="lds01", MODE="0666"
```

## 3. ESP32 firmware

ESP32-S3 firmware 需自製:
- Arduino IDE 或 PlatformIO
- WiFi scan API + USB CDC serial output
- Output 格式(每個 scan event 一行 JSON):
  ```json
  {
    "scan_duration_ms": 3801,
    "aps": [
      {"ssid":"BMELab","bssid":"14:DA:E9:80:B0:34","rssi":-63,"channel":1,"encryption":"WPA2"},
      ...
    ]
  }
  ```

**本 repo 不含 ESP32 firmware 原始碼**(外部依賴)。任何符合上述格式的 firmware 都可。

## 4. SSH 從 VM 免密登入 Pi

```bash
# VM
ssh-keygen
ssh-copy-id wayne@wayne.local
# 之後 ssh 不再問密碼
```

## 5. 跑起來

### 5.1 建圖階段

```bash
# VM
ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py
# 內含:Pi bringup(via ssh)+ Cartographer + RViz
```

### 5.2 蒐集 WiFi 階段

```bash
# VM
ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py
# 內含:Pi bringup + map_server + amcl + wifi_collector + RViz
# RViz 拉 2D Pose Estimate → 推 30cm → 遙控走 15-30 分鐘
```

### 5.3 後處理

```bash
cd ~/ros2_ws/lab2_configs
python3 jsonl_to_csv.py --split-by-time 30
python3 make_rssi_heatmap.py
python3 make_morning_evening_diff.py    # 需 2026-05-17 + 2026-05-23 jsonl
python3 make_trajectory_overlay.py
```

---

## 疑難排解(實際遇過)

| 症狀 | 根因 | 解法 |
|---|---|---|
| 跨機 `/tf /scan` 收不到 | 預設 FastRTPS 在 WiFi 上 multicast 不可靠 | 改 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| SLAM 沉默地漂掉 | VirtualBox 時鐘漂移 | 裝 `chrony`,VM 跟 Pi 對時;漂太兇 `sudo chronyc -a makestep` |
| `/dev/ttyUSB0` 找不到 | USB 裝置編號不固定 | 用 udev rules 固定 `/dev/lds01` 等 symlink |
| hlds_laser_publisher 卡住 | LDS-01 馬達沒轉 | 檢查 OpenCR 電源開關 + 5V 線 + 電池電壓 |
| AMCL `TF map→base_footprint not ready` | 還沒拉 2D Pose Estimate | RViz 點 2D Pose Estimate 給 initial pose |
| data_collector 配到 3 秒前 pose | `lookup_transform` 用 latest TF | 用 `Time.from_msg(msg.header.stamp)` lookup TF at scan time |
