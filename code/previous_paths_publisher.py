#!/usr/bin/env python3
"""previous_paths_publisher — show all already-walked trajectories in RViz.

Reads trajectories_slim.csv (already cleaned + path_id assigned) and publishes
all rows as a single MarkerArray on /previous_paths so you can see in RViz
what's been covered before. Each path_id gets a distinct HSV color (matches
what jsonl_to_csv.py emits).

Usage:
  python3 ~/ros2_ws/lab2_configs/previous_paths_publisher.py
  python3 ~/ros2_ws/lab2_configs/previous_paths_publisher.py --ros-args \\
      -p csv_path:=/home/wayne/ros2_ws/wifi_data/trajectories_slim.csv

In RViz, add a MarkerArray display on topic /previous_paths.
"""
import colorsys
import csv
import os
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class PreviousPathsPublisher(Node):
    def __init__(self):
        super().__init__('previous_paths_publisher')

        self.declare_parameter('csv_path',
            os.path.expanduser('~/ros2_ws/wifi_data/trajectories_slim.csv'))
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('line_width', 0.06)
        self.declare_parameter('publish_period_sec', 5.0)

        self.csv_path = self.get_parameter('csv_path').value
        self.frame_id = self.get_parameter('frame_id').value
        self.line_width = float(self.get_parameter('line_width').value)

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(MarkerArray, '/previous_paths', latched)

        self.marker_array = self._build_marker_array()
        self.pub.publish(self.marker_array)
        self.get_logger().info(
            f'Loaded {len(self.marker_array.markers)} path markers from {self.csv_path}')

        # Periodically re-publish so RViz subscribers attaching late get it
        self.create_timer(float(self.get_parameter('publish_period_sec').value),
                          self._republish)

    def _republish(self):
        # Update header stamps so RViz keeps them alive
        now = self.get_clock().now().to_msg()
        for m in self.marker_array.markers:
            m.header.stamp = now
        self.pub.publish(self.marker_array)

    def _build_marker_array(self) -> MarkerArray:
        arr = MarkerArray()
        if not os.path.exists(self.csv_path):
            self.get_logger().warn(f'CSV not found: {self.csv_path}')
            return arr

        # group rows by path_id
        paths = defaultdict(list)   # pid -> [(x, y)]
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            for row in reader:
                try:
                    pid = int(row[0])
                    x   = float(row[1])
                    y   = float(row[2])
                except (ValueError, IndexError):
                    continue
                paths[pid].append((x, y))

        n_paths = max(paths) if paths else 1
        now = self.get_clock().now().to_msg()
        for pid in sorted(paths.keys()):
            pts = paths[pid]
            if len(pts) < 2:
                continue
            h = (pid - 1) / max(1, n_paths)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = 'previous_paths'
            m.id = pid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = self.line_width
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            m.lifetime.sec = 0   # forever
            for x, y in pts:
                p = Point(); p.x = float(x); p.y = float(y); p.z = 0.05
                m.points.append(p)
            arr.markers.append(m)
        return arr


def main():
    rclpy.init()
    node = PreviousPathsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
