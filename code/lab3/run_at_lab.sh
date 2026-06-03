#!/usr/bin/env bash
# 實驗室現場 demo 一鍵啟動。
#
# 用法:
#     ./run_at_lab.sh                 # Live (ESP32),5-seed ensemble(最準)
#     ./run_at_lab.sh single          # Live,只跑 seed 42 — 箭頭永遠在 heatmap 上
#     ./run_at_lab.sh replay          # Replay 用內建 jsonl
#     ./run_at_lab.sh replay single   # Replay + single seed
#     ./run_at_lab.sh replay /path/to/some.jsonl
#
# 自動處理:
#   - source ROS 2 Humble + ROS_DOMAIN_ID=30
#   - 檢查 /dev/ttyACM0 存在 + 可讀,不行就提示 sudo chmod
#   - 一條 launch 把 localizer + RViz 起來

set -e
cd "$(dirname "$0")"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Accept `single` as either the first or second arg
unset WIFI_NO_ENSEMBLE
for arg in "$@"; do
    [ "$arg" = "single" ] && export WIFI_NO_ENSEMBLE=1
done
if [ -n "$WIFI_NO_ENSEMBLE" ]; then
    echo "[mode] single-seed (arrow on heatmap, ~0.1 m worse than ensemble)"
fi

# Strip `single` from positional args so MODE/JSONL still work
ARGS=()
for arg in "$@"; do
    [ "$arg" != "single" ] && ARGS+=("$arg")
done
set -- "${ARGS[@]}"

MODE="${1:-live}"
JSONL="${2:-../../wifi/wifi_20260523_231102.jsonl}"

if [ "$MODE" = "live" ]; then
    PORT="/dev/ttyACM0"
    if [ ! -e "$PORT" ]; then
        echo "ERR: $PORT not found.  Check ESP32 USB passthrough in VirtualBox."
        echo "    lsusb | grep -i espressif"
        exit 1
    fi
    if [ ! -r "$PORT" ] || [ ! -w "$PORT" ]; then
        echo "ERR: $PORT not readable.  Run once:"
        echo "    sudo chmod 666 $PORT"
        exit 1
    fi
    echo "[ok] $PORT ready, launching..."
    exec ros2 launch ./lab3_demo.launch.py mode:=live port:=$PORT smooth:=5 min_aps:=2
elif [ "$MODE" = "replay" ]; then
    if [ ! -f "$JSONL" ]; then
        echo "ERR: jsonl not found: $JSONL"
        exit 1
    fi
    echo "[ok] replay $JSONL"
    exec ros2 launch ./lab3_demo.launch.py mode:=replay jsonl:="$(realpath $JSONL)" smooth:=5 min_aps:=2 interval:=0.5
else
    echo "usage: $0 [live|replay] [jsonl_path]"
    exit 1
fi
