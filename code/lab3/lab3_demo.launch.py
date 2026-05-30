"""Lab 3 即時定位 demo — 一條指令啟動 localizer + RViz.

預設 Live 模式(從 /dev/ttyACM0 讀 ESP32)。要 Replay 改參數即可:
    ros2 launch lab3_demo.launch.py mode:=replay jsonl:=/path/to.jsonl

Live(預設):
    ros2 launch lab3_demo.launch.py
    ros2 launch lab3_demo.launch.py port:=/dev/ttyACM1   # 換 port
    ros2 launch lab3_demo.launch.py smooth:=10           # 站著不動更穩
    ros2 launch lab3_demo.launch.py min_aps:=3           # 少於 3 個 matched 就 LOW

Replay(不需 ESP32):
    ros2 launch lab3_demo.launch.py mode:=replay jsonl:=../../wifi/wifi_20260523_231102.jsonl
    ros2 launch lab3_demo.launch.py mode:=replay jsonl:=... interval:=0.3

關閉 RViz(只跑 publisher,等別人接):
    ros2 launch lab3_demo.launch.py rviz:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    here = os.path.dirname(os.path.realpath(__file__))
    node_py = os.path.join(here, 'esp32_localizer_ros.py')
    rviz_cfg = os.path.join(here, 'esp32_localizer.rviz')
    default_jsonl = os.path.realpath(
        os.path.join(here, '..', '..', 'wifi', 'wifi_20260523_231102.jsonl'))

    mode = LaunchConfiguration('mode')
    port = LaunchConfiguration('port')
    baud = LaunchConfiguration('baud')
    smooth = LaunchConfiguration('smooth')
    min_aps = LaunchConfiguration('min_aps')
    jsonl = LaunchConfiguration('jsonl')
    interval = LaunchConfiguration('interval')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='live',
            choices=['live', 'replay'],
            description='live: read ESP32 from --port; replay: step through a jsonl'),
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0',
            description='ESP32 serial port (live mode)'),
        DeclareLaunchArgument('baud', default_value='115200'),
        DeclareLaunchArgument('smooth', default_value='5',
            description='Moving-average over last N predictions (1 = raw)'),
        DeclareLaunchArgument('min_aps', default_value='2',
            description='Mark LOW CONFIDENCE if fewer known APs matched'),
        DeclareLaunchArgument('jsonl', default_value=default_jsonl,
            description='jsonl file path for replay mode'),
        DeclareLaunchArgument('interval', default_value='0.5',
            description='Seconds between replay scans'),
        DeclareLaunchArgument('rviz', default_value='true',
            description='Also start RViz'),

        # Live mode
        ExecuteProcess(
            condition=LaunchConfigurationEquals('mode', 'live'),
            cmd=['python3', node_py,
                 '--port', port,
                 '--baud', baud,
                 '--smooth', smooth,
                 '--min-aps', min_aps],
            output='screen',
            name='wifi_localizer',
        ),

        # Replay mode
        ExecuteProcess(
            condition=LaunchConfigurationEquals('mode', 'replay'),
            cmd=['python3', node_py,
                 '--replay', jsonl,
                 '--replay-interval', interval,
                 '--smooth', smooth,
                 '--min-aps', min_aps],
            output='screen',
            name='wifi_localizer',
        ),

        # RViz
        ExecuteProcess(
            condition=IfCondition(rviz),
            cmd=['rviz2', '-d', rviz_cfg, '--ros-args', '--log-level', 'WARN'],
            output='screen',
            name='rviz2',
        ),
    ])
