#!/usr/bin/env python3
"""auto_trajectory_node — fully automatic trajectory generator for Lab2.

Subscribes /map, picks N goal poses in free space (Poisson-disk style), sends
them to nav2 via NavigateToPose action one at a time, and publishes a
/current_path_id Int32 (latched) so external nodes / post-processing can tag
which wifi sample belongs to which path.

On exit (or after all goals done) writes path_metadata.json with:
  { path_id : { start_wall, end_wall, color_rgb, goal_xy, status } }

Usage:
  python3 ~/ros2_ws/lab2_configs/auto_trajectory_node.py --ros-args \\
      -p n_trajectories:=20 \\
      -p min_goal_separation_m:=1.5 \\
      -p wall_clearance_m:=0.4 \\
      -p nav_timeout_per_goal_sec:=90.0 \\
      -p metadata_path:=/home/wayne/ros2_ws/wifi_data/path_metadata.json

Requires:
  * lab2.launch.py running (nav2 active, slam_toolbox publishing /map)
  * NavigateToPose action server up
"""
import colorsys
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, ColorRGBA
from nav2_msgs.action import NavigateToPose


def hsv_color(i: int, n: int) -> tuple:
    """Evenly distribute hue across N paths so colors are distinct."""
    h = (i % n) / n
    s, v = 0.85, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def poisson_disk_sample(free_cells: np.ndarray, n_target: int,
                        min_dist_cells: float, max_attempts: int = 5000):
    """Bridson-ish Poisson-disk sampling on a list of candidate (cx, cy) cells.

    Returns up to n_target points well separated by min_dist_cells.
    """
    if len(free_cells) == 0:
        return []
    chosen = []
    attempts = 0
    rng = random.Random(42)
    min_dist_sq = min_dist_cells * min_dist_cells
    # Random first seed
    chosen.append(tuple(free_cells[rng.randrange(len(free_cells))]))
    while len(chosen) < n_target and attempts < max_attempts:
        cand = tuple(free_cells[rng.randrange(len(free_cells))])
        ok = True
        for c in chosen:
            dx = cand[0] - c[0]; dy = cand[1] - c[1]
            if dx*dx + dy*dy < min_dist_sq:
                ok = False
                break
        if ok:
            chosen.append(cand)
        attempts += 1
    return chosen


def nearest_neighbor_order(points: list, start_xy: tuple = (0.0, 0.0)) -> list:
    """Greedy TSP-ish ordering: keep visiting the nearest unvisited point."""
    remaining = list(points)
    ordered = []
    cur = start_xy
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0]-cur[0])**2 + (p[1]-cur[1])**2)
        ordered.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return ordered


class AutoTrajectoryNode(Node):
    def __init__(self):
        super().__init__('auto_trajectory_node')

        self.declare_parameter('n_trajectories', 20)
        self.declare_parameter('min_goal_separation_m', 1.5)
        self.declare_parameter('wall_clearance_m', 0.4)
        self.declare_parameter('nav_timeout_per_goal_sec', 90.0)
        # Per lab spec "軌跡為 T 秒 (ex. 30秒)" — each path's wifi-recording
        # window is at most this long. After T sec we advance to next goal
        # even if nav2 hasn't reached it yet.
        self.declare_parameter('traj_duration_sec', 30.0)
        self.declare_parameter('metadata_path',
            os.path.expanduser('~/ros2_ws/wifi_data/path_metadata.json'))
        self.declare_parameter('settle_seconds_per_goal', 2.0)

        self.n_target = int(self.get_parameter('n_trajectories').value)
        self.min_sep_m = float(self.get_parameter('min_goal_separation_m').value)
        self.wall_clear_m = float(self.get_parameter('wall_clearance_m').value)
        self.nav_timeout = float(self.get_parameter('nav_timeout_per_goal_sec').value)
        self.traj_T = float(self.get_parameter('traj_duration_sec').value)
        self.metadata_path = self.get_parameter('metadata_path').value
        self.settle = float(self.get_parameter('settle_seconds_per_goal').value)
        # Effective timeout per goal = min(nav_timeout, traj_T)
        # so T (spec param) acts as a hard upper bound on each trajectory
        self.effective_timeout = min(self.nav_timeout, self.traj_T)

        # /current_path_id — latched / transient_local so late subscribers (e.g.
        # wifi taggers started after we're already on path 5) immediately get
        # the current value rather than waiting for the next change.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.path_pub = self.create_publisher(Int32, '/current_path_id', latched)
        self.color_pub = self.create_publisher(ColorRGBA, '/current_path_color', latched)

        # /map — slam_toolbox publishes with transient_local so we get last msg
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map: OccupancyGrid = None
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # path_id 0 reserved for "before auto-traj started / idle"
        self._publish_path_id(0, (0.5, 0.5, 0.5))

        self.metadata = {}      # path_id -> dict
        self.goals_xy = []      # list of (x, y) in map frame
        self.current_pid = 0
        self.started = False

        self.get_logger().info(
            f'Auto-trajectory ready. waiting for /map then {self.n_target} goals.')
        self._kickoff_timer = self.create_timer(2.0, self._maybe_kickoff)

    def _publish_path_id(self, pid: int, rgb: tuple):
        msg = Int32(); msg.data = pid; self.path_pub.publish(msg)
        c = ColorRGBA(); c.r, c.g, c.b = rgb; c.a = 1.0
        self.color_pub.publish(c)

    def _on_map(self, msg: OccupancyGrid):
        if self.map is None:
            self.get_logger().info(
                f'/map received: {msg.info.width}×{msg.info.height} @ '
                f'{msg.info.resolution:.3f} m/cell, origin '
                f'({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})')
        self.map = msg

    def _maybe_kickoff(self):
        if self.started or self.map is None:
            return
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('navigate_to_pose action server not ready yet...')
            return
        self.started = True
        self._kickoff_timer.cancel()
        self._sample_goals_from_map()
        if not self.goals_xy:
            self.get_logger().error('Could not sample any free goal — aborting.')
            rclpy.shutdown()
            return
        self.get_logger().info(f'sampled {len(self.goals_xy)} goals; starting nav loop.')
        self._next_goal()

    # ------------------------------------------------------------------ goal pool
    def _sample_goals_from_map(self):
        """Use occupancy grid to find free cells, dilate occupied to enforce
        wall clearance, then Poisson-disk subsample to n_target goals."""
        mp = self.map
        res = mp.info.resolution
        W, H = mp.info.width, mp.info.height
        ox, oy = mp.info.origin.position.x, mp.info.origin.position.y

        grid = np.array(mp.data, dtype=np.int16).reshape(H, W)
        # 0 = free, 100 = occupied, -1 = unknown
        free = (grid == 0)
        occ = (grid > 50)

        # Dilate occupied by wall_clearance to keep goals away from walls
        clearance_cells = max(1, int(round(self.wall_clear_m / res)))
        try:
            from scipy.ndimage import binary_dilation
            occ_d = binary_dilation(occ, iterations=clearance_cells)
            safe = free & ~occ_d
        except Exception:
            self.get_logger().warn('scipy not available; using raw free cells')
            safe = free

        ys, xs = np.where(safe)
        if len(xs) == 0:
            return
        # convert cell idx → (x_map, y_map) center of cell
        # OccupancyGrid: cell (cx, cy) center is at (origin + (cx+0.5)*res, origin + (cy+0.5)*res)
        # but we store as cell coords for the poisson distance check
        cells = np.column_stack([xs, ys])
        min_dist_cells = self.min_sep_m / res
        picked_cells = poisson_disk_sample(cells, self.n_target, min_dist_cells)

        goals_world = [(ox + (cx + 0.5) * res, oy + (cy + 0.5) * res)
                       for (cx, cy) in picked_cells]
        goals_world = nearest_neighbor_order(goals_world, start_xy=(0.0, 0.0))
        self.goals_xy = goals_world

    # ------------------------------------------------------------------ nav loop
    def _next_goal(self):
        if not self.goals_xy:
            self._all_done()
            return
        gx, gy = self.goals_xy.pop(0)
        self.current_pid += 1
        color = hsv_color(self.current_pid - 1, self.n_target)
        self.metadata[self.current_pid] = {
            'goal_xy': [gx, gy],
            'color_rgb': list(color),
            'start_wall': time.time(),
            'end_wall': None,
            'status': 'IN_PROGRESS',
        }
        self._publish_path_id(self.current_pid, color)
        self.get_logger().info(
            f'─── path_id={self.current_pid}/{self.n_target}  goal=({gx:.2f},{gy:.2f})  '
            f'color={color}')

        # Build PoseStamped facing the direction of travel (yaw = atan2 from current pos)
        # We don't know current pos here so just use yaw=0; nav2 doesn't care for nav-to-pose
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.w = 1.0  # identity quaternion

        # Async send + register both feedback and result callbacks
        self._send_future = self.nav_client.send_goal_async(goal)
        self._send_future.add_done_callback(self._on_goal_accepted)
        # Watchdog timer in case goal hangs forever
        self._timeout_timer = self.create_timer(self.nav_timeout, self._on_goal_timeout)

    def _on_goal_accepted(self, future):
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn(f'path_id={self.current_pid} REJECTED by nav2')
            self.metadata[self.current_pid]['status'] = 'REJECTED'
            self.metadata[self.current_pid]['end_wall'] = time.time()
            self._timeout_timer.cancel()
            self._next_goal()
            return
        self._goal_handle = gh
        result_future = gh.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        self._timeout_timer.cancel()
        res = future.result()
        status = res.status if res else -1
        # nav2 GoalStatus: 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
        status_str = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, f'STATUS_{status}')
        self.get_logger().info(f'path_id={self.current_pid} → {status_str}')
        self.metadata[self.current_pid]['status'] = status_str
        self.metadata[self.current_pid]['end_wall'] = time.time()
        # Brief settle so wifi_collector finishes any in-flight sample at goal
        time.sleep(self.settle)
        self._next_goal()

    def _on_goal_timeout(self):
        self._timeout_timer.cancel()
        self.get_logger().warn(
            f'path_id={self.current_pid} TIMEOUT after {self.nav_timeout}s — cancelling')
        try:
            self._goal_handle.cancel_goal_async()
        except Exception:
            pass
        self.metadata[self.current_pid]['status'] = 'TIMEOUT'
        self.metadata[self.current_pid]['end_wall'] = time.time()
        # _on_goal_result will fire shortly after cancel completes

    def _all_done(self):
        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info(f'  all {self.current_pid} paths processed')
        self.get_logger().info(f'  metadata → {self.metadata_path}')
        self._publish_path_id(-1, (0.0, 0.0, 0.0))  # sentinel: finished
        self._write_metadata()

    def _write_metadata(self):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        meta = {
            'n_target': self.n_target,
            'n_completed': self.current_pid,
            'wall_clearance_m': self.wall_clear_m,
            'min_goal_separation_m': self.min_sep_m,
            'paths': self.metadata,
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'  wrote {len(self.metadata)} path records')


def main():
    rclpy.init()
    node = AutoTrajectoryNode()

    def handle_sigint(sig, frame):
        node.get_logger().warn('SIGINT — saving metadata before quit')
        try:
            node._write_metadata()
        except Exception as e:
            node.get_logger().error(f'metadata write failed: {e}')
        rclpy.shutdown()
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._write_metadata()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
