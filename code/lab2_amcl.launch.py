"""lab2_amcl.launch.py — pure AMCL localization on pre-built map, NO SLAM.

This is what you want when collecting wifi data on a map that's already
finalized. No scan-matching to a growing map, no Cartographer optimization,
just particle-filter localization that lights up your robot pose on the map.

What it starts:
  Pi (via ssh):  turtlebot3 bringup + twist_mux
  VM (locally):  map_server (publishes psquare as /map)
                 amcl       (particle filter; outputs map→odom TF)
                 lifecycle_manager (activates the above)
                 RViz       (for 2D Pose Estimate + visual confirmation)
                 wifi_collector on Pi (auto-starts at 20s)

How AMCL localizes:
  1. You publish /initialpose (RViz "2D Pose Estimate" button does this).
  2. AMCL scatters a bunch of particles around that pose.
  3. As /scan + /odom come in, particles weighted by how well their
     predicted scan matches the static map; bad particles die.
  4. After 5–10 seconds of driving, particles converge to true pose.
  5. /tf publishes map → odom continuously; wifi_collector reads it for pose.

Usage:
  ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py
  ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py map_yaml:=...
  ros2 launch ~/ros2_ws/lab2_configs/lab2_amcl.launch.py enable_wifi:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PI_BRINGUP_CMD = (
    'source /opt/ros/humble/setup.bash && '
    'source ~/turtlebot3_ws/install/setup.bash && '
    'source ~/ros2_ws/install/setup.bash && '
    'export ROS_DOMAIN_ID=30 && '
    'export TURTLEBOT3_MODEL=burger && '
    'export LDS_MODEL=LDS-01 && '
    'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
    'ros2 launch ~/ros2_ws/maps/pi_robot_all.launch.py'
)

PI_WIFI_CMD = (
    'source /opt/ros/humble/setup.bash && '
    'source ~/ros2_ws/install/setup.bash && '
    'export ROS_DOMAIN_ID=30 && '
    'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
    'ros2 launch wifi_collector collect.launch.py'
)


def generate_launch_description():
    configs = os.path.expanduser('~/ros2_ws/lab2_configs')
    maps    = os.path.expanduser('~/ros2_ws/maps')
    nav2_params = os.path.join(configs, 'nav2_burger.yaml')   # has amcl section
    rviz_config = os.path.join(configs, 'lab2.rviz')

    map_yaml = LaunchConfiguration('map_yaml')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    start_pi = LaunchConfiguration('start_pi')
    rviz = LaunchConfiguration('rviz')
    enable_wifi = LaunchConfiguration('enable_wifi')
    pi_host = LaunchConfiguration('pi_host')

    localization_launch = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch', 'localization_launch.py')

    return LaunchDescription([
        DeclareLaunchArgument('map_yaml',
            default_value=os.path.join(maps, 'psquare.yaml'),
            description='Map yaml (must reference a .pgm in same dir)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_pi',    default_value='true'),
        DeclareLaunchArgument('pi_host',     default_value='wayne@wayne.local'),
        DeclareLaunchArgument('rviz',        default_value='true'),
        DeclareLaunchArgument('enable_wifi', default_value='true'),

        # ── Pi bringup via ssh ───────────────────────────────────────────
        ExecuteProcess(
            condition=IfCondition(start_pi),
            cmd=['ssh', '-tt', '-o', 'StrictHostKeyChecking=accept-new',
                 pi_host, PI_BRINGUP_CMD],
            output='screen', name='pi_bringup_ssh',
            sigterm_timeout='4', sigkill_timeout='6',
        ),

        # ── AMCL + map_server (nav2 standard localization stack) ────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(localization_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'map': map_yaml,
                'params_file': nav2_params,
                'autostart': 'true',
            }.items(),
        ),

        # ── RViz ─────────────────────────────────────────────────────────
        Node(
            condition=IfCondition(rviz),
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'WARN'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            sigterm_timeout='3', sigkill_timeout='5',
        ),

        # ── wifi_collector on Pi (wait 20s for AMCL to come up) ─────────
        TimerAction(
            period=20.0,
            actions=[ExecuteProcess(
                condition=IfCondition(enable_wifi),
                cmd=['ssh', '-tt', '-o', 'StrictHostKeyChecking=accept-new',
                     pi_host, PI_WIFI_CMD],
                output='screen', name='pi_wifi_collector_ssh',
            )],
        ),

        # ── Previous-paths visualizer ────────────────────────────────────
        # Reads existing trajectories_slim.csv and publishes all 49 paths as a
        # latched MarkerArray on /previous_paths so RViz shows what you've
        # already walked → drive new areas to maximize coverage.
        ExecuteProcess(
            cmd=['python3', os.path.expanduser(
                '~/ros2_ws/lab2_configs/previous_paths_publisher.py')],
            output='screen', name='previous_paths_publisher',
        ),
    ])
