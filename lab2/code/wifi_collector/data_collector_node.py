"""Pair each incoming WifiScan with the latest robot pose and append to a JSONL file.

Event-driven by design: every WifiScan triggers one record. Pose is looked up
from TF at the moment the scan arrives — never timer-driven. A record is dropped
if TF isn't ready yet or if the AP count is below min_aps.
"""
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from my_interface.msg import WifiScan


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        self.declare_parameter('output_dir', os.path.expanduser('~/ros2_ws/wifi_data'))
        self.declare_parameter('output_basename', '')  # empty = auto timestamp
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('min_aps', 1)
        self.declare_parameter('require_tf', True)

        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        self.min_aps = int(self.get_parameter('min_aps').value)
        self.require_tf = bool(self.get_parameter('require_tf').value)

        out_dir = self.get_parameter('output_dir').value
        os.makedirs(out_dir, exist_ok=True)
        basename = self.get_parameter('output_basename').value
        if not basename:
            basename = time.strftime('wifi_%Y%m%d_%H%M%S')
        self.out_path = os.path.join(out_dir, basename + '.jsonl')

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.sub = self.create_subscription(WifiScan, 'wifi_scan', self._on_scan, 10)

        self._file = open(self.out_path, 'a', encoding='utf-8')
        self._n_saved = 0
        self._n_dropped_tf = 0
        self._n_dropped_minaps = 0

        self.get_logger().info(f'data_collector_node writing to {self.out_path}')

    def destroy_node(self):
        if self._file:
            self._file.close()
        super().destroy_node()

    def _on_scan(self, msg: WifiScan):
        if len(msg.aps) < self.min_aps:
            self._n_dropped_minaps += 1
            return

        pose = None
        try:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            try:
                tf = self.buffer.lookup_transform(
                    self.source_frame, self.target_frame, stamp)
            except (LookupException, ConnectivityException, ExtrapolationException):
                # Fallback to latest TF if scan stamp is too old for TF buffer
                tf = self.buffer.lookup_transform(
                    self.source_frame, self.target_frame, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            pose = {
                'frame_id': self.source_frame,
                'x': t.x, 'y': t.y, 'z': t.z,
                'yaw': quat_to_yaw(q.x, q.y, q.z, q.w),
                'quat': {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w},
            }
        except (LookupException, ConnectivityException, ExtrapolationException):
            if self.require_tf:
                self._n_dropped_tf += 1
                if self._n_dropped_tf % 10 == 1:
                    self.get_logger().warn(
                        f'TF {self.source_frame}->{self.target_frame} not ready; '
                        f'dropped {self._n_dropped_tf} scans so far')
                return

        record = {
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'wall_time': time.time(),
            'pose': pose,
            'scan_duration_ms': msg.scan_duration_ms,
            'aps': [{
                'ssid': ap.ssid,
                'bssid': ap.bssid,
                'rssi': ap.rssi,
                'channel': ap.channel,
                'encryption': ap.encryption,
            } for ap in msg.aps],
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._file.flush()
        self._n_saved += 1
        if self._n_saved % 10 == 1:
            self.get_logger().info(
                f'saved={self._n_saved} dropped_tf={self._n_dropped_tf} '
                f'dropped_minaps={self._n_dropped_minaps}')


def main():
    rclpy.init()
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
