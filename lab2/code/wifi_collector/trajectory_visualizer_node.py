"""Publish accumulated trajectory as nav_msgs/Path for RViz overlay on the map."""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class TrajectoryVisualizerNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer_node')

        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('sample_rate_hz', 5.0)
        self.declare_parameter('min_dist_m', 0.05)
        self.declare_parameter('max_points', 5000)

        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame = self.get_parameter('source_frame').value
        rate = float(self.get_parameter('sample_rate_hz').value)
        self.min_dist = float(self.get_parameter('min_dist_m').value)
        self.max_points = int(self.get_parameter('max_points').value)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(Path, 'trajectory', 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)

        self.path = Path()
        self.path.header.frame_id = self.source_frame
        self._last_xy = None

    def _tick(self):
        try:
            tf = self.buffer.lookup_transform(
                self.source_frame, self.target_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        if self._last_xy is not None:
            dx = x - self._last_xy[0]
            dy = y - self._last_xy[1]
            if (dx * dx + dy * dy) ** 0.5 < self.min_dist:
                return
        self._last_xy = (x, y)

        ps = PoseStamped()
        ps.header.stamp = tf.header.stamp
        ps.header.frame_id = self.source_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = tf.transform.translation.z
        ps.pose.orientation = tf.transform.rotation

        self.path.poses.append(ps)
        if len(self.path.poses) > self.max_points:
            self.path.poses = self.path.poses[-self.max_points:]
        self.path.header.stamp = tf.header.stamp
        self.pub.publish(self.path)


def main():
    rclpy.init()
    node = TrajectoryVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
