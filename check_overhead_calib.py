#!/usr/bin/env python3
"""
check_overhead_calib.py - live check of the overhead camera's intrinsic calibration.

Subscribes to the RAW image + camera_info, finds a checkerboard (default 12x8 inner
corners = 13x9 squares, 30 mm squares; change with --board / --square), fits the board pose with solvePnP using the
loaded calibration (K + distortion), and prints the reprojection RMS error in pixels.

Low error (< ~0.5 px) on NEW board positions, including near the image edges,
means the loaded calibration is good.

Usage:
  python3 check_overhead_calib.py
  python3 check_overhead_calib.py --image /overhead/image_raw --info /overhead/camera_info
  python3 check_overhead_calib.py --board 9x6 --square 0.025      # inner corners, metres
Ctrl+C prints a summary.
"""
import argparse
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

DETECT_SCALE = 0.5        # detect on a half-size image for speed, refine at full size


class CalibCheck(Node):
    def __init__(self, image_topic, info_topic, cols, rows, square):
        super().__init__('check_overhead_calib')
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.errors = []
        self.frame = 0
        self.size = (cols, rows)
        obj = np.zeros((rows * cols, 3), np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
        self.obj = obj
        self.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f'Waiting for {info_topic} and {image_topic} ...')

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(
                f'Loaded calibration: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}')

    def on_image(self, msg):
        if self.K is None:
            return
        self.frame += 1
        if self.frame % 5:          # check every 5th frame to keep CPU load low
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        found, corners = cv2.findChessboardCorners(
            small, self.size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if not found:
            print('board not found (whole board must be visible)', end='\r')
            return
        corners = corners / DETECT_SCALE
        corners = cv2.cornerSubPix(
            gray, corners.astype(np.float32), (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        ok, rvec, tvec = cv2.solvePnP(self.obj, corners, self.K, self.D)
        if not ok:
            return
        proj, _ = cv2.projectPoints(self.obj, rvec, tvec, self.K, self.D)
        err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
        self.errors.append(err)
        cx, cy = corners.reshape(-1, 2).mean(axis=0)
        dist = float(np.linalg.norm(tvec))
        verdict = 'good' if err < 0.5 else ('ok' if err < 1.0 else 'HIGH')
        print(f'RMS {err:5.2f} px [{verdict:4}]  board centre at pixel ({cx:6.0f},{cy:5.0f})  '
              f'distance {dist:.2f} m              ')

    def summary(self):
        if not self.errors:
            print('\nNo board detections.')
            return
        e = np.array(self.errors)
        print(f'\n{len(e)} detections: mean {e.mean():.2f} px, median {np.median(e):.2f} px, '
              f'max {e.max():.2f} px')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/overhead/image_raw')
    ap.add_argument('--info', default='/overhead/camera_info')
    ap.add_argument('--board', default='12x8', help='INNER corners, cols x rows')
    ap.add_argument('--square', type=float, default=0.030, help='square side, metres')
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    cols, rows = (int(v) for v in args.board.lower().split('x'))
    node = CalibCheck(args.image, args.info, cols, rows, args.square)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
