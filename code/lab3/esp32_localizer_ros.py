"""ROS 2 real-time WiFi localizer node — ESP32 on VM, RViz visualisation.

Loads the Cascade 5-seed ensemble (`outputs/checkpoints/A_random__Cascade_s4[2-6].pt`),
reads ESP32 scan JSON from serial, publishes:

  /map                  nav_msgs/OccupancyGrid    psquare.pgm (latched)
  /wifi_heatmap         nav_msgs/OccupancyGrid    model probability over 40×33 grid (latched)
  /wifi_estimated_pose  geometry_msgs/PoseStamped predicted (x,y)
  /wifi_confidence      visualization_msgs/Marker text overlay (n_matched APs)

Usage:
    # one terminal
    source /opt/ros/humble/setup.bash
    cd ~/lab2_submit_FINAL_20260523/code/lab3
    python3 esp32_localizer_ros.py --port /dev/ttyACM0

    # another terminal
    source /opt/ros/humble/setup.bash
    rviz2 -d esp32_localizer.rviz

Replay mode (no hardware) for sanity check:
    python3 esp32_localizer_ros.py --replay ../../wifi/wifi_20260517_101315.jsonl
"""
import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

import data
import models


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CKPT_DIR = HERE / 'outputs' / 'checkpoints'
BSSID_JSON = HERE / 'outputs' / 'bssids.json'
MAP_YAML = REPO_ROOT / 'map' / 'psquare.yaml'

SEEDS = [42, 43, 44, 45, 46]
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
MAX_APS = 50
RSSI_MISSING = -100.0


class Localizer:
    """Holds the 5-seed Cascade ensemble; maps a scan list → (xy, heatmap, n_matched)."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.bssids = json.load(open(BSSID_JSON, encoding='utf-8'))
        self.bidx = {b.upper(): i for i, b in enumerate(self.bssids)}

        self.fine_xy, self.Gw, self.Gh = data.build_heatmap_grid(cell_size=0.4)
        fine_mask = data.build_free_mask(self.fine_xy)
        coarse_xy, _, _ = data.build_heatmap_grid(cell_size=1.6)
        coarse_mask = data.build_free_mask(coarse_xy)
        f2c = data.build_fine_to_coarse(fine_cell=0.4, coarse_cell=1.6)
        self.bounds = data.HEATMAP_BOUNDS

        self.models = []
        for s in SEEDS:
            ck = CKPT_DIR / f'A_random__Cascade_s{s}.pt'
            if not ck.exists():
                continue
            m = models.SetTransformerHeatmapCascade(
                num_bssids=len(self.bssids),
                fine_cell_xy=self.fine_xy, fine_free_mask=fine_mask.astype(np.float32),
                coarse_cell_xy=coarse_xy, coarse_free_mask=coarse_mask.astype(np.float32),
                fine_to_coarse=f2c, **CFG).to(self.device)
            m.load_state_dict(torch.load(ck, map_location=self.device))
            m.eval()
            self.models.append(m)
        if not self.models:
            raise RuntimeError('No Cascade checkpoints found in outputs/checkpoints/')
        print(f'[ok] loaded {len(self.models)} seed models')

    def _scan_to_input(self, aps):
        idx = np.full((1, MAX_APS), len(self.bssids), dtype=np.int64)
        val = np.zeros((1, MAX_APS), dtype=np.float32)
        mask = np.zeros((1, MAX_APS), dtype=np.float32)
        ranked = sorted(
            ((str(a.get('bssid', '')).upper(), float(a.get('rssi', RSSI_MISSING)))
             for a in aps if str(a.get('bssid', '')).upper() in self.bidx),
            key=lambda kv: kv[1], reverse=True)[:MAX_APS]
        n_matched = len(ranked)
        for j, (b, v) in enumerate(ranked):
            idx[0, j] = self.bidx[b]
            val[0, j] = (v - RSSI_MISSING) / 20.0
            mask[0, j] = 1.0
        return (torch.from_numpy(idx), torch.from_numpy(val),
                torch.from_numpy(mask), n_matched)

    @torch.no_grad()
    def localize(self, aps):
        it, vt, mt, n_matched = self._scan_to_input(aps)
        xys, probs = [], []
        for m in self.models:
            fine_l, coarse_l = m(it, vt, mt)
            p = m._gated_fine_prob(fine_l, coarse_l)
            probs.append(p.cpu().numpy()[0])
            xys.append((p @ m.fine_xy).cpu().numpy()[0])
        return np.mean(xys, axis=0), np.mean(probs, axis=0).reshape(self.Gh, self.Gw), n_matched


# ────────────────────────── ROS node ───────────────────────────────────
class LocalizerNode(Node):
    def __init__(self, args):
        super().__init__('esp32_localizer')
        self.args = args
        self.loc = Localizer(device=args.device)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.pub_map = self.create_publisher(OccupancyGrid, '/map', latched)
        self.pub_heat = self.create_publisher(OccupancyGrid, '/wifi_heatmap', latched)
        self.pub_pose = self.create_publisher(PoseStamped, '/wifi_estimated_pose', 10)
        self.pub_text = self.create_publisher(Marker, '/wifi_confidence', 10)

        self._publish_map_once()

        self.smooth_buf = deque(maxlen=max(1, args.smooth))
        self.seq = 0

    # ── load + publish floor plan ──
    def _publish_map_once(self):
        with open(MAP_YAML) as f:
            mp = yaml.safe_load(f)
        pgm = np.array(Image.open(MAP_YAML.parent / mp['image']))
        H, W = pgm.shape
        msg = OccupancyGrid()
        msg.header = Header(frame_id='map')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = MapMetaData(
            resolution=float(mp['resolution']),
            width=W, height=H,
        )
        msg.info.origin.position.x = float(mp['origin'][0])
        msg.info.origin.position.y = float(mp['origin'][1])
        msg.info.origin.orientation.w = 1.0
        # nav2 occupancy: 0=free, 100=occupied, -1=unknown
        data = np.zeros(H * W, dtype=np.int8)
        flat = np.flipud(pgm).reshape(-1)        # ROS origin = bottom-left
        data[flat >= 200] = 0
        data[flat < 100] = 100
        data[(flat >= 100) & (flat < 200)] = -1
        msg.data = data.tolist()
        self.pub_map.publish(msg)
        self.get_logger().info(f'published map {W}x{H} @ {mp["resolution"]} m/cell')

    # ── inference callback ──
    def on_scan(self, aps):
        n = len(aps)
        try:
            xy, heat, n_matched = self.loc.localize(aps)
        except Exception as e:                                  # noqa: BLE001
            self.get_logger().error(f'inference failed: {e}')
            return

        low_conf = n_matched < self.args.min_aps
        if not low_conf:
            self.smooth_buf.append(xy)
            xy_smooth = np.mean(self.smooth_buf, axis=0)
        else:
            xy_smooth = xy   # don't pollute buffer

        stamp = self.get_clock().now().to_msg()
        self.seq += 1

        # PoseStamped
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = 'map'
        ps.pose.position.x = float(xy_smooth[0])
        ps.pose.position.y = float(xy_smooth[1])
        ps.pose.orientation.w = 1.0
        self.pub_pose.publish(ps)

        # Heatmap OccupancyGrid
        self._publish_heatmap(heat, stamp)

        # Confidence text
        self._publish_confidence(xy_smooth, n_matched, n, low_conf, stamp)

        flag = ' LOW' if low_conf else ''
        self.get_logger().info(
            f'#{self.seq:5d}  {n_matched:>2d}/{n:>2d} APs  →  '
            f'({xy_smooth[0]:+.2f}, {xy_smooth[1]:+.2f}) m{flag}')

    def _publish_heatmap(self, heat, stamp):
        # heat: (Gh, Gw) in HEATMAP_BOUNDS frame
        Gh, Gw = heat.shape
        bx0, _, by0, _ = self.loc.bounds
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.info.resolution = 0.4
        msg.info.width = Gw
        msg.info.height = Gh
        msg.info.origin.position.x = float(bx0)
        msg.info.origin.position.y = float(by0)
        msg.info.origin.orientation.w = 1.0
        # normalize to [0, 100]
        h = heat.astype(np.float32)
        if h.max() > 0:
            h = h / h.max() * 100.0
        # ROS expects row-major from origin (bottom-left); our heat is (Gh, Gw)
        # with row 0 = bottom of bounds (build_heatmap_grid uses y outer/inner increasing)
        msg.data = h.flatten().astype(np.int8).tolist()
        self.pub_heat.publish(msg)

    def _publish_confidence(self, xy, n_matched, n_total, low_conf, stamp):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'map'
        m.ns = 'confidence'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        if low_conf:
            m.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            m.text = f'LOW CONF\n{n_matched}/{n_total} APs'
        else:
            m.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=1.0)
            m.text = f'{n_matched}/{n_total} APs'
        self.pub_text.publish(m)


# ────────────────────────── scan sources ──────────────────────────────
def replay_thread(node, path, interval):
    with open(path, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip().startswith('{')]
    node.get_logger().info(f'replaying {len(lines)} scans from {path}, interval={interval}s')
    for ln in lines:
        if not rclpy.ok():
            return
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        node.on_scan(d.get('aps', []))
        time.sleep(interval)
    node.get_logger().info('replay done')


def serial_thread(node, port, baud):
    import serial
    buf = b''
    while rclpy.ok():
        try:
            with serial.Serial(port, baud, timeout=1.0) as ser:
                node.get_logger().info(f'serial open: {port} @ {baud}')
                while rclpy.ok():
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b'\n' in buf:
                        raw, buf = buf.split(b'\n', 1)
                        s = raw.decode('utf-8', errors='replace').strip()
                        if not s.startswith('{'):
                            continue
                        try:
                            d = json.loads(s)
                        except json.JSONDecodeError:
                            continue
                        node.on_scan(d.get('aps', []))
        except Exception as e:                                  # noqa: BLE001
            node.get_logger().warn(f'serial error: {e}; retry in 2s')
            time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--port', help='ESP32 serial port, e.g. /dev/ttyACM0')
    grp.add_argument('--replay', help='Replay scans from a jsonl file (no hardware)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--smooth', type=int, default=1,
                    help='Moving average over last N predictions (default 1 = raw)')
    ap.add_argument('--min-aps', type=int, default=2,
                    help='If fewer known APs matched, mark LOW CONFIDENCE (red)')
    ap.add_argument('--replay-interval', type=float, default=1.0,
                    help='Seconds between replay scans')
    args = ap.parse_args()

    rclpy.init()
    node = LocalizerNode(args)

    if args.port:
        t = threading.Thread(target=serial_thread, args=(node, args.port, args.baud),
                             daemon=True)
    else:
        t = threading.Thread(target=replay_thread,
                             args=(node, args.replay, args.replay_interval), daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
