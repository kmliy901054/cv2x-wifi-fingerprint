"""CV2X Lab2 — minimal VM-side launch.

Architecture:
  Pi  (via ssh): turtlebot3 bringup + twist_mux         (pi_robot_all.launch.py)
  VM (locally): Cartographer (no-IMU) + nav2 + RViz

CycloneDDS replaces FastRTPS so /tf flows cross-machine — no relay needed.

Usage:
    ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py
    ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py start_pi:=false   # Pi already up
    ros2 launch ~/ros2_ws/lab2_configs/lab2.launch.py rviz:=false       # headless

Manual extras (separate terminals):
    keyboard teleop:
        ros2 run turtlebot3_teleop teleop_keyboard --ros-args -r /cmd_vel:=/cmd_vel_teleop
    wifi data collection (on Pi when ready to record):
        ssh wayne@wayne.local "bash -ic 'ros2 launch wifi_collector collect.launch.py'"

Save a map:
    ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/maps/lab2_$(date +%Y%m%d_%H%M)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


PI_REMOTE_CMD = (
    'source /opt/ros/humble/setup.bash && '
    'source ~/turtlebot3_ws/install/setup.bash && '
    'source ~/ros2_ws/install/setup.bash && '
    'export ROS_DOMAIN_ID=30 && '
    'export TURTLEBOT3_MODEL=burger && '
    'export LDS_MODEL=LDS-01 && '
    'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
    'ros2 launch ~/ros2_ws/maps/pi_robot_all.launch.py'
)


def generate_launch_description():
    configs = os.path.expanduser('~/ros2_ws/lab2_configs')
    nav2_params = os.path.join(configs, 'nav2_burger.yaml')
    rviz_config = os.path.join(configs, 'lab2.rviz')
    carto_lua = configs                          # directory holding .lua
    carto_lua_basename = 'cartographer_lab2.lua'

    nav2_launch = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch', 'navigation_launch.py')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    start_pi = LaunchConfiguration('start_pi')
    rviz = LaunchConfiguration('rviz')
    pi_host = LaunchConfiguration('pi_host')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_pi', default_value='true',
                              description='SSH to Pi and start bringup+twist_mux'),
        DeclareLaunchArgument('pi_host', default_value='wayne@wayne.local',
                              description='SSH target for the Pi'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with the lab2 preset'),

        # ── Pi side (via SSH; -tt forces a PTY so Ctrl-C propagates) ─────
        ExecuteProcess(
            condition=IfCondition(start_pi),
            cmd=['ssh', '-tt', '-o', 'StrictHostKeyChecking=accept-new',
                 pi_host, PI_REMOTE_CMD],
            output='screen',
            name='pi_bringup_ssh',
            sigterm_timeout='4', sigkill_timeout='6',
        ),

        # ── SLAM (Cartographer per lab spec) ─────────────────────────────
        # Matches the lab's "ros2 launch turtlebot3_cartographer cartographer.launch.py"
        # but uses our custom no-IMU lua tuned for the venue.
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory', carto_lua,
                '-configuration_basename',  carto_lua_basename,
            ],
            remappings=[('/imu', '/imu_unused')],  # belt-and-suspenders: ignore /imu
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'resolution': 0.05,
                         'publish_period_sec': 1.0}],
        ),

        # ── Nav2 (cmd_vel → /cmd_vel_nav so Pi twist_mux can pick teleop;
        #         cmd_vel_smoothed → DROP so velocity_smoother stops fighting twist_mux) ─
        GroupAction([
            SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
            SetRemap(src='/cmd_vel_smoothed', dst='/cmd_vel_nav_smoothed_DROP'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'params_file': nav2_params,
                    'autostart': 'true',
                }.items(),
            ),
        ]),

        # ── RViz ─────────────────────────────────────────────────────────
        # log-level WARN silences the cosmetic "Lookup would require extrapolation"
        # spam from the Map / LaserScan displays (root cause: some msg stamp gets
        # interpreted as 2006 — RViz-side issue, doesn't affect SLAM/nav).
        # Bump back to INFO if debugging RViz itself.
        Node(
            condition=IfCondition(rviz),
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'WARN'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            sigterm_timeout='3', sigkill_timeout='5',
        ),
    ])
