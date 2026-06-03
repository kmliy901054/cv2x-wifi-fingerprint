"""Ground-truth error helper.

Listens to:
  /initialpose             (RViz "2D Pose Estimate" — your actual position)
  /wifi_estimated_pose     (the localizer's prediction)

For every scan after a GT click, prints:
    err = |pred - gt|   m
and publishes /wifi_gt_marker (a green sphere at your clicked spot).

Run alongside the localizer:
    python3 gt_error_node.py
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


class GTNode(Node):
    def __init__(self):
        super().__init__('wifi_gt_error')
        self.gt = None
        self.errors = []

        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.on_gt, 10)
        self.create_subscription(
            PointStamped, '/clicked_point', self.on_gt_point, 10)
        self.create_subscription(
            PoseStamped, '/wifi_estimated_pose', self.on_pred, 10)
        self.pub_marker = self.create_publisher(Marker, '/wifi_gt_marker', 10)
        self.get_logger().info(
            'ready — RViz "Publish Point" (click) or "2D Pose Estimate" (drag) to mark your position')

    def on_gt_point(self, msg):
        # Same handler logic as PoseWithCovarianceStamped, but for a PointStamped
        class _FakePose:                        # tiny shim so on_gt's API stays
            class pose:
                class pose:
                    pass
        fake = _FakePose()
        fake.pose.pose.position = msg.point
        self.on_gt(fake)

    def on_gt(self, msg):
        self.gt = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.errors.clear()
        self.get_logger().info(
            f'GT set → ({self.gt[0]:+.2f}, {self.gt[1]:+.2f}) m. '
            f'Will log error on each new prediction.')
        self._publish_marker()

    def on_pred(self, msg):
        if self.gt is None:
            return
        pred = (msg.pose.position.x, msg.pose.position.y)
        e = math.hypot(pred[0] - self.gt[0], pred[1] - self.gt[1])
        self.errors.append(e)
        n = len(self.errors)
        med = float(np.median(self.errors))
        avg = float(np.mean(self.errors))
        self.get_logger().info(
            f'pred ({pred[0]:+.2f}, {pred[1]:+.2f})  '
            f'GT ({self.gt[0]:+.2f}, {self.gt[1]:+.2f})  '
            f'err={e:.2f} m   '
            f'[n={n}  med={med:.2f}  mean={avg:.2f}]')

    def _publish_marker(self):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'gt'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.gt[0])
        m.pose.position.y = float(self.gt[1])
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=0.9)
        self.pub_marker.publish(m)


def main():
    rclpy.init()
    n = GTNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
