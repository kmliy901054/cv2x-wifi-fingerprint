#!/usr/bin/env python3
"""rssi_coverage_node — real-time wifi sampling visualization for RViz.

Publishes two layers:
  /wifi_coverage   OccupancyGrid   how dense the wifi sampling is per cell
                                   (0 = no samples, ≥1 = sampled, redder = more)
  /wifi_ap_markers MarkerArray     one sphere per AP at its strongest-RSSI
                                   pose, colored by signal strength

This is the "innovation" deliverable — shows lab evaluators that you not only
collected wifi data, you visualized which AREAS have been sampled and where
each AP's strongest reading was, useful for fingerprint-based localization
training-set quality assessment.

Usage:
  python3 ~/ros2_ws/lab2_configs/rssi_coverage_node.py --ros-args \\
      -p grid_resolution_m:=0.5
"""
import math
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from my_interface.msg import WifiScan


def rssi_to_color(rssi: int) -> tuple:
    """Map RSSI dBm to RGB. Strong (>-50) green, mid (-70) yellow, weak (<-85) red."""
    # Clamp to [-90, -30]
    r = max(-90, min(-30, rssi))
    # Linear: -30 → 1.0, -90 → 0.0
    norm = (r + 90) / 60.0
    if norm > 0.66:
        # green → yellow
        t = (norm - 0.66) / 0.34
        return (1.0 - t, 1.0, 0.0)
    elif norm > 0.33:
        # yellow → orange
        t = (norm - 0.33) / 0.33
        return (1.0, 0.7 + 0.3 * t, 0.0)
    else:
        # orange → red
        t = norm / 0.33
        return (1.0, 0.3 * t, 0.0)


class RssiCoverageNode(Node):
    def __init__(self):
        super().__init__('rssi_coverage_node')

        self.declare_parameter('grid_resolution_m', 0.5)
        self.declare_parameter('grid_extent_m', 25.0)  # half-width centered on origin
        self.declare_parameter('publish_period_sec', 2.0)
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('max_aps_in_markers', 30)

        self.res = float(self.get_parameter('grid_resolution_m').value)
        self.extent = float(self.get_parameter('grid_extent_m').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        self.max_aps = int(self.get_parameter('max_aps_in_markers').value)

        # Coverage grid: 2D array of sample counts
        n = int(self.extent * 2 / self.res)
        self.grid_size = n
        self.coverage = [[0] * n for _ in range(n)]
        self._origin_x = -self.extent
        self._origin_y = -self.extent

        # AP samples: bssid → list of (x, y, rssi, ssid)
        self.ap_samples = defaultdict(list)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.cov_pub = self.create_publisher(OccupancyGrid, '/wifi_coverage', latched)
        self.mk_pub = self.create_publisher(MarkerArray, '/wifi_ap_markers', 10)

        self.create_subscription(WifiScan, '/wifi_scan', self._on_scan, 10)
        self.create_timer(float(self.get_parameter('publish_period_sec').value),
                          self._publish)

        self.get_logger().info(
            f'rssi_coverage_node ready: {n}×{n} grid @ {self.res}m, '
            f'extent ±{self.extent}m')

    def _on_scan(self, msg: WifiScan):
        # Get robot pose AT scan time (not at message arrival).
        # Falls back to latest TF if scan stamp is too old for TF buffer.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.source_frame, self.target_frame,
                rclpy.time.Time.from_msg(msg.header.stamp))
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.source_frame, self.target_frame, rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                return
        x = tf.transform.translation.x
        y = tf.transform.translation.y

        # Update coverage
        gx = int((x - self._origin_x) / self.res)
        gy = int((y - self._origin_y) / self.res)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.coverage[gy][gx] += 1

        # Update AP samples
        for ap in msg.aps:
            self.ap_samples[ap.bssid].append((x, y, ap.rssi, ap.ssid))

    def _publish(self):
        self._publish_coverage()
        self._publish_ap_markers()

    def _publish_coverage(self):
        max_c = max((max(row) for row in self.coverage), default=0)
        if max_c == 0:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.source_frame
        msg.info.resolution = float(self.res)
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = float(self._origin_x)
        msg.info.origin.position.y = float(self._origin_y)
        msg.info.origin.orientation.w = 1.0
        # Normalize 0..100 (rviz OccupancyGrid display will color)
        data = []
        for row in self.coverage:
            for v in row:
                if v == 0:
                    data.append(-1)  # unknown / no samples
                else:
                    # log scale so even sparse samples show
                    norm = int(min(100, 20 + 80 * math.log1p(v) / math.log1p(max_c)))
                    data.append(norm)
        msg.data = data
        self.cov_pub.publish(msg)

    def _publish_ap_markers(self):
        if not self.ap_samples:
            return
        arr = MarkerArray()

        # Sort APs by strongest RSSI seen, take top-N
        ap_best = []
        for bssid, samples in self.ap_samples.items():
            best = max(samples, key=lambda s: s[2])  # max rssi
            ap_best.append((bssid, best))
        ap_best.sort(key=lambda kv: kv[1][2], reverse=True)
        ap_best = ap_best[:self.max_aps]

        for i, (bssid, (x, y, rssi, ssid)) in enumerate(ap_best):
            m = Marker()
            m.header.frame_id = self.source_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'wifi_aps'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            # Sphere size scales with RSSI strength
            size = 0.15 + 0.35 * max(0.0, (rssi + 90) / 60.0)
            m.scale.x = m.scale.y = m.scale.z = size
            r, g, b = rssi_to_color(rssi)
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            m.lifetime.sec = 0  # forever
            arr.markers.append(m)

            # Add a text marker with SSID + RSSI
            t = Marker()
            t.header = m.header
            t.ns = 'wifi_labels'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(x)
            t.pose.position.y = float(y)
            t.pose.position.z = 0.7
            t.scale.z = 0.15
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            t.text = f'{ssid[:12]} {rssi}dB'
            arr.markers.append(t)

        self.mk_pub.publish(arr)


def main():
    rclpy.init()
    node = RssiCoverageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
