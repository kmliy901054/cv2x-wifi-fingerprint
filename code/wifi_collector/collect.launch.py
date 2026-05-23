"""Launch all 4 wifi_collector nodes with shared params yaml.

Run with e.g.:
    ros2 launch wifi_collector collect.launch.py

Override params on CLI:
    ros2 launch wifi_collector collect.launch.py port:=/dev/ttyUSB0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('wifi_collector')
    default_params = os.path.join(pkg_share, 'config', 'collector.yaml')

    params_file = LaunchConfiguration('params_file')
    port = LaunchConfiguration('port')
    source_frame = LaunchConfiguration('source_frame')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Path to collector params yaml'),
        DeclareLaunchArgument('port', default_value='/dev/esp32',
                              description='ESP32 serial port'),
        DeclareLaunchArgument('source_frame', default_value='map',
                              description='Fixed frame to look up pose against (map or odom)'),

        Node(
            package='wifi_collector', executable='esp32_bridge_node',
            name='esp32_bridge_node', output='screen',
            parameters=[params_file, {'port': port}],
        ),
        Node(
            package='wifi_collector', executable='pose_tracker_node',
            name='pose_tracker_node', output='screen',
            parameters=[params_file, {'source_frame': source_frame}],
        ),
        Node(
            package='wifi_collector', executable='data_collector_node',
            name='data_collector_node', output='screen',
            parameters=[params_file, {'source_frame': source_frame}],
        ),
        Node(
            package='wifi_collector', executable='trajectory_visualizer_node',
            name='trajectory_visualizer_node', output='screen',
            parameters=[params_file, {'source_frame': source_frame}],
        ),
    ])
