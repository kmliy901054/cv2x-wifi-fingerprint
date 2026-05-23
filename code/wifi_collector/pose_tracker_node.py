"""Track latest robot pose by polling TF, publish as PoseStamped.

The data_collector_node could query TF directly, but having a dedicated node
keeps that logic in one place and lets us also republish the pose at a steady
rate for visualization / debugging.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class PoseTrackerNode(Node):
    def __init__(self):
        super().__init__('pose_tracker_node')

        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        rate = float(self.get_parameter('publish_rate_hz').value)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(PoseStamped, 'robot_pose', 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)

        self._warned = False
        self.get_logger().info(
            f'pose_tracker_node: looking up {self.source_frame} -> {self.target_frame}')

    def _tick(self):
        try:
            tf = self.buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            if not self._warned:
                self.get_logger().warn(f'TF not ready yet ({e}); will keep retrying silently')
                self._warned = True
            return

        if self._warned:
            self.get_logger().info('TF ready.')
            self._warned = False

        msg = PoseStamped()
        msg.header.stamp = tf.header.stamp
        msg.header.frame_id = self.source_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PoseTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
