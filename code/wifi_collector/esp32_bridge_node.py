"""Read ESP32 Wi-Fi scan output from serial and publish as my_interface/WifiScan.

Expected ESP32 firmware emits one JSON object per line, e.g.:
  {"scan_duration_ms": 3421, "aps": [{"ssid":"NTU-WiFi","bssid":"aa:bb:cc:dd:ee:ff",
                                       "rssi":-65,"ch":6,"enc":"WPA2"}, ...]}

Lines that fail to parse are dropped with a warning. One WifiScan message is
published per valid line — this is what makes the pipeline event-driven.
"""
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import serial

from my_interface.msg import WifiAP, WifiScan


class Esp32BridgeNode(Node):
    def __init__(self):
        super().__init__('esp32_bridge_node')

        self.declare_parameter('port', '/dev/ttyACM1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'esp32_wifi')
        self.declare_parameter('reopen_delay_sec', 2.0)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.reopen_delay = float(self.get_parameter('reopen_delay_sec').value)

        self.pub = self.create_publisher(WifiScan, 'wifi_scan', 10)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'esp32_bridge_node started: port={self.port} baud={self.baudrate}')

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().destroy_node()

    def _reader_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            try:
                with serial.Serial(self.port, self.baudrate, timeout=1.0) as ser:
                    self.get_logger().info(f'Serial opened: {self.port}')
                    buf = b''
                    while not self._stop.is_set() and rclpy.ok():
                        chunk = ser.read(256)
                        if not chunk:
                            continue
                        buf += chunk
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            self._handle_line(line.decode('utf-8', errors='replace').strip())
            except serial.SerialException as e:
                self.get_logger().warn(f'Serial error: {e}; retrying in {self.reopen_delay}s')
                self._stop.wait(self.reopen_delay)

    def _handle_line(self, line: str):
        if not line or not line.startswith('{'):
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.get_logger().debug(f'skip non-JSON: {line[:80]}')
            return

        msg = WifiScan()
        # Scan took ~3 seconds; stamp should be the SCAN MID-POINT, not message
        # arrival time. Backdate by half the scan duration so downstream nodes
        # pair the wifi with where the robot WAS during the scan, not after.
        scan_dur_ms = float(payload.get('scan_duration_ms', 0.0))
        now = self.get_clock().now()
        backdated = now - Duration(nanoseconds=int(scan_dur_ms * 1e6 / 2))
        msg.header.stamp = backdated.to_msg()
        msg.header.frame_id = self.frame_id
        msg.scan_duration_ms = scan_dur_ms

        for ap in payload.get('aps', []):
            a = WifiAP()
            a.ssid = str(ap.get('ssid', ''))
            a.bssid = str(ap.get('bssid', ''))
            a.rssi = int(ap.get('rssi', 0))
            a.channel = int(ap.get('ch', ap.get('channel', 0))) & 0xFF
            a.encryption = str(ap.get('enc', ap.get('encryption', '')))
            msg.aps.append(a)

        self.pub.publish(msg)
        self.get_logger().info(
            f'WifiScan: {len(msg.aps)} APs, scan_dur={msg.scan_duration_ms:.0f}ms')


def main():
    rclpy.init()
    node = Esp32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
