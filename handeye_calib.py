#!/usr/bin/env python3
"""
handeye_calib.py - eye-in-hand calibration for a wrist-mounted camera (e.g. RealSense).

Finds T_tool0<-camera_optical: where the wrist camera sits relative to the robot flange.

Three modes:

  1) identify  - work out the ChArUco board's dictionary and layout from the live image
       python3 handeye_calib.py identify
       python3 handeye_calib.py identify --file board.png

  2) capture   - jog the arm to 15-20 varied poses looking at a FIXED board, press keys:
       python3 handeye_calib.py capture --dict DICT_5X5_100 --squares 6x8 \
           --square 0.040 --marker 0.030
     or with a printed AprilTag/ArUco grid (e.g. 5 x 7 tags, ids 0..34 row by row from the
     top-left, 20 mm black squares, 30.5 mm centre-to-centre pitch; measure, don't trust the PDF):
       python3 handeye_calib.py capture --target grid --dict DICT_APRILTAG_36h11 \
           --grid 5x7 --tag 0.020 --pitch-x 0.0305 --pitch-y 0.0305
     Keys in the image window:
       c = capture sample (robot must be STOPPED)   u = undo last sample
       s = solve and print result                    w = write samples to file
       q = quit (also writes samples)

  3) solve     - re-solve offline from a saved samples file
       python3 handeye_calib.py solve handeye_samples.json

Requirements: ROS 2 (rclpy, tf2_ros, cv_bridge), OpenCV with the aruco module,
UR driver running (publishes base_link -> tool0), realsense2_camera running.
"""
import argparse
import json
import sys

import numpy as np
import cv2

ARUCO = cv2.aruco

DICT_NAMES = [
    'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
    'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
    'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
    'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
    'DICT_ARUCO_ORIGINAL',
    'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11',
]

HE_METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ----------------------------------------------------------------------------- OpenCV compatibility
# Ubuntu 22.04 apt OpenCV is 4.5.x (old aruco API); pip/newer builds are 4.7+ (new API).

def get_dict(name):
    did = getattr(ARUCO, name)
    if hasattr(ARUCO, 'getPredefinedDictionary'):
        return ARUCO.getPredefinedDictionary(did)
    return ARUCO.Dictionary_get(did)


def detect_markers(gray, dictionary, subpix=False):
    """subpix=True refines marker corners to sub-pixel accuracy (used for pose estimation)."""
    if hasattr(ARUCO, 'ArucoDetector'):
        params = ARUCO.DetectorParameters()
        if subpix:
            params.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
        det = ARUCO.ArucoDetector(dictionary, params)
        corners, ids, _ = det.detectMarkers(gray)
    else:
        params = ARUCO.DetectorParameters_create()
        if subpix:
            params.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
        corners, ids, _ = ARUCO.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids


def make_board(sx, sy, square, marker, dictionary, legacy=False):
    if hasattr(ARUCO, 'CharucoBoard_create'):          # OpenCV <= 4.6
        return ARUCO.CharucoBoard_create(sx, sy, square, marker, dictionary)
    board = ARUCO.CharucoBoard((sx, sy), square, marker, dictionary)
    if legacy and hasattr(board, 'setLegacyPattern'):   # boards printed with old OpenCV
        board.setLegacyPattern(True)
    return board


def board_corners_3d(board):
    if hasattr(board, 'getChessboardCorners'):
        return np.array(board.getChessboardCorners(), np.float32).reshape(-1, 3)
    return np.array(board.chessboardCorners, np.float32).reshape(-1, 3)


def detect_charuco(gray, board, dictionary):
    if hasattr(ARUCO, 'CharucoDetector'):
        det = ARUCO.CharucoDetector(board)
        ch_corners, ch_ids, m_corners, m_ids = det.detectBoard(gray)
    else:
        m_corners, m_ids = detect_markers(gray, dictionary)
        ch_corners, ch_ids = None, None
        if m_ids is not None and len(m_ids) > 0:
            _, ch_corners, ch_ids = ARUCO.interpolateCornersCharuco(
                m_corners, m_ids, gray, board)
    return ch_corners, ch_ids, m_corners, m_ids


def charuco_pose(ch_corners, ch_ids, board, K, D, min_corners=8):
    if ch_ids is None or ch_corners is None or len(ch_ids) < min_corners:
        return None
    obj = board_corners_3d(board)[np.asarray(ch_ids).flatten()]
    img = np.asarray(ch_corners, np.float32).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return rvec, tvec, err, len(obj)


def grid_object_points(cols, rows, tag, pitch_x, pitch_y, first_id=0):
    """AprilTag/ArUco grid: id = first_id + row*cols + col, row 0 at the top.
    Board frame: origin at tag first_id's top-left outer corner, x right, y down, z into board.
    Corner order matches OpenCV detection: TL, TR, BR, BL."""
    pts = {}
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * pitch_x, r * pitch_y
            pts[first_id + r * cols + c] = np.array(
                [[x0, y0, 0], [x0 + tag, y0, 0], [x0 + tag, y0 + tag, 0], [x0, y0 + tag, 0]],
                np.float32)
    return pts


def grid_pose(gray, dictionary, grid_pts, K, D, min_tags=4, rotation=None):
    """rotation: how the printed tags are turned relative to OpenCV's corner order
    (0/1/2/3 quarter turns). Some tag generators print 36h11 tags rotated vs OpenCV, so
    by default all four are tried and the one with the lowest reprojection error wins.
    Returns ((rvec, tvec, err, n_tags, rotation) or None, marker_corners, marker_ids)."""
    m_corners, m_ids = detect_markers(gray, dictionary, subpix=True)
    if m_ids is None:
        return None, m_corners, m_ids
    obj_tags, img = [], []
    for c, i in zip(m_corners, m_ids.flatten()):
        if int(i) in grid_pts:
            obj_tags.append(grid_pts[int(i)])
            img.append(np.asarray(c, np.float32).reshape(4, 2))
    if len(obj_tags) < min_tags:
        return None, m_corners, m_ids
    img = np.concatenate(img)
    best = None
    for rot in ([rotation] if rotation is not None else [0, 1, 2, 3]):
        obj = np.concatenate([np.roll(t, -rot, axis=0) for t in obj_tags])
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if best is None or err < best[2]:
            best = (rvec, tvec, err, len(obj_tags), rot)
    return best, m_corners, m_ids


def draw_axes(img, K, D, rvec, tvec, length):
    if hasattr(cv2, 'drawFrameAxes'):
        cv2.drawFrameAxes(img, K, D, rvec, tvec, length)
    else:
        ARUCO.drawAxis(img, K, D, rvec, tvec, length)


# ----------------------------------------------------------------------------- transform helpers

def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def tf_msg_to_mat(t):
    q, p = t.transform.rotation, t.transform.translation
    M = np.eye(4)
    M[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    M[:3, 3] = [p.x, p.y, p.z]
    return M


def rt_to_mat(rvec, tvec):
    M = np.eye(4)
    M[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    M[:3, 3] = np.asarray(tvec, float).flatten()
    return M


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


# ----------------------------------------------------------------------------- solve

def _solve_all(T_bt, T_cb):
    """Run every OpenCV hand-eye method; return {name: (X, per_sample_dev_m, rot_spread_deg)}."""
    results = {}
    for name, method in HE_METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(
                [T[:3, :3] for T in T_bt], [T[:3, 3] for T in T_bt],
                [T[:3, :3] for T in T_cb], [T[:3, 3] for T in T_cb], method=method)
        except cv2.error as e:
            print(f'{name}: failed ({e})')
            continue
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = R, t.flatten()
        # The FIXED board's pose in the base frame should come out the same for every sample.
        T_bb = [Tb @ X @ Tc for Tb, Tc in zip(T_bt, T_cb)]
        pos = np.array([T[:3, 3] for T in T_bb])
        dev = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
        rot = [rot_angle_deg(T_bb[0][:3, :3].T @ T[:3, :3]) for T in T_bb]
        results[name] = (X, dev, float(np.mean(rot)))
    return results


def solve(samples, cam_link_T=None, frames=('tool0', 'camera_color_optical_frame'),
          reject=True):
    """samples: list of dicts with T_base_tool (4x4) and rvec/tvec (board in camera).
    With reject=True, samples that disagree badly with the rest are dropped and the
    calibration is re-solved (they are listed so you can see which poses were bad)."""
    if len(samples) < 3:
        print('Need at least 3 samples (15-20 recommended).')
        return None
    if not hasattr(cv2, 'calibrateHandEye'):
        print(f'This OpenCV ({cv2.__version__}) has no calibrateHandEye. Use the system OpenCV 4.x '
              '(python3-opencv) or: pip install opencv-contrib-python==4.10.0.84')
        return None
    T_bt_all = [np.array(s['T_base_tool']) for s in samples]
    T_cb_all = [rt_to_mat(s['rvec'], s['tvec']) for s in samples]

    max_rot = max(rot_angle_deg(T_bt_all[0][:3, :3].T @ T[:3, :3]) for T in T_bt_all)
    if max_rot < 15:
        print(f'WARNING: wrist orientations span only {max_rot:.1f} deg. '
              'Rotate the wrist more (aim for 20-40 deg in several directions).')

    keep = list(range(len(samples)))
    dropped = []
    while True:
        res = _solve_all([T_bt_all[i] for i in keep], [T_cb_all[i] for i in keep])
        if not res:
            return None
        best = min(res, key=lambda k: np.mean(res[k][1]))
        dev = res[best][1]
        med = float(np.median(dev))
        worst = int(np.argmax(dev))
        # Drop the worst sample if it is far off the rest; keep at least 8 (or 2/3 of them).
        if (reject and dev[worst] > max(0.005, 4 * med)
                and len(keep) > max(8, int(np.ceil(2 * len(samples) / 3)))):
            dropped.append((keep[worst] + 1, dev[worst] * 1000))
            keep.pop(worst)
            continue
        break

    print('\nPer-sample check (board position vs the median, best method):')
    for j, i in enumerate(keep):
        s = samples[i]
        flag = '  <-- suspicious' if dev[j] > max(0.003, 3 * med) else ''
        print(f'  sample {i + 1:2d}: {dev[j] * 1000:6.2f} mm   reproj {s.get("reproj_px", float("nan")):.2f}px'
              f'{flag}')
    if dropped:
        print('Dropped as outliers: ' + ', '.join(f'sample {n} ({d:.0f} mm off)' for n, d in dropped))

    print('\nMethod        board-position spread (mean / max)   board-rotation spread')
    for name, (_, d, r) in res.items():
        print(f'  {name:11s}   {np.mean(d) * 1000:6.2f} mm / {np.max(d) * 1000:6.2f} mm'
              f'                {r:5.2f} deg')
    X = res[best][0]
    q = R_to_quat(X[:3, :3])
    print(f'\nBest: {best} using {len(keep)} of {len(samples)} samples.  '
          f'T_{frames[0]} <- {frames[1]}:')
    print(f'  xyz  = [{X[0,3]:.5f}, {X[1,3]:.5f}, {X[2,3]:.5f}]  (m)')
    print(f'  quat = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]  (x, y, z, w)')
    print('  (camera offset from flange: '
          f'{np.linalg.norm(X[:3, 3]) * 1000:.1f} mm; sanity-check against the mount)')
    m = float(np.mean(dev)) * 1000
    verdict = 'good' if m < 2 else ('usable' if m < 5 else 'poor: add more varied poses')
    print(f'  Fixed-board consistency {m:.2f} mm mean -> {verdict}')

    if cam_link_T is not None:
        # RealSense already publishes camera_link -> ..._optical_frame; publish tool0 -> camera_link.
        T_link_opt = np.array(cam_link_T)
        Y = X @ np.linalg.inv(T_link_opt)
        qy = R_to_quat(Y[:3, :3])
        print('\nFor the TF tree (RealSense publishes camera_link -> optical itself), use:')
        print('  ros2 run tf2_ros static_transform_publisher '
              f'--x {Y[0,3]:.5f} --y {Y[1,3]:.5f} --z {Y[2,3]:.5f} '
              f'--qx {qy[0]:.6f} --qy {qy[1]:.6f} --qz {qy[2]:.6f} --qw {qy[3]:.6f} '
              f'--frame-id {frames[0]} --child-frame-id camera_link')
    return X


# ----------------------------------------------------------------------------- identify (offline helper)

def identify_image(gray, try_sizes):
    scores = []
    for name in DICT_NAMES:
        d = get_dict(name)
        _, ids = detect_markers(gray, d)
        n = 0 if ids is None else len(ids)
        if n:
            scores.append((n, name, sorted(int(i) for i in ids.flatten())))
    if not scores:
        print('No markers detected with any dictionary. Get the whole board in view, sharp and well lit.')
        return
    scores.sort(key=lambda s: (-s[0], DICT_NAMES.index(s[1])))
    print('Dictionaries that detected markers (best first):')
    for n, name, ids in scores[:4]:
        print(f'  {name:22s} {n:3d} markers, ids {ids[0]}..{ids[-1]}')
    n, best, ids = scores[0]
    print(f'\n-> Most likely dictionary: {best}  ({n} markers visible, highest id {ids[-1]})')

    d = get_dict(best)
    print('\nBoard layout check (more ChArUco corners = right layout):')
    for sx, sy in try_sizes:
        for legacy in (False, True):
            if legacy and hasattr(ARUCO, 'CharucoBoard_create'):
                continue  # old OpenCV has no legacy switch
            board = make_board(sx, sy, 1.0, 0.75, d, legacy)
            ch_c, ch_i, _, _ = detect_charuco(gray, board, d)
            k = 0 if ch_i is None else len(ch_i)
            print(f'  --squares {sx}x{sy}{" --legacy" if legacy else "":9s} -> {k:3d} corners '
                  f'(max possible {(sx - 1) * (sy - 1)})')
    print('\nThen measure with calipers: one square side (--square, metres) and one marker side '
          '(--marker, metres).')


# ----------------------------------------------------------------------------- ROS node

def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import tf2_ros

    class HandEye(Node):
        def __init__(self):
            super().__init__('handeye_calib')
            self.bridge = CvBridge()
            self.K = self.D = None
            self.img_msg = None
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_subscription(CameraInfo, args.info, self.on_info, 10)
            self.create_subscription(Image, args.image, self.on_image, 10)

        def on_info(self, msg):
            if self.K is None:
                self.K = np.array(msg.k, float).reshape(3, 3)
                self.D = np.array(msg.d, float)
                self.optical_frame = msg.header.frame_id or 'camera_color_optical_frame'
                self.get_logger().info(f'Camera info received (frame {self.optical_frame}).')

        def on_image(self, msg):
            self.img_msg = msg

        def lookup(self, parent, child):
            try:
                t = self.tf_buffer.lookup_transform(parent, child, Time())
                return tf_msg_to_mat(t)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'TF {parent} -> {child} not available: {e}')
                return None

    rclpy.init()
    node = HandEye()
    node.get_logger().info(f'Waiting for {args.image} and {args.info} ...')

    if args.mode == 'identify':
        while rclpy.ok() and node.img_msg is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        gray = cv2.cvtColor(node.bridge.imgmsg_to_cv2(node.img_msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        identify_image(gray, args.try_sizes)
        node.destroy_node()
        rclpy.try_shutdown()
        return

    d = get_dict(args.dict)
    if args.target == 'charuco':
        sx, sy = args.squares
        board = make_board(sx, sy, args.square, args.marker, d, args.legacy)
        axis_len = args.square * 3
    else:
        gc, gr = args.grid
        grid_pts = grid_object_points(gc, gr, args.tag, args.pitch_x, args.pitch_y, args.first_id)
        axis_len = args.pitch_x * 2
    samples, cam_link_T = [], None
    cv2.namedWindow('handeye', cv2.WINDOW_NORMAL)
    print('Keys: c=capture (robot stopped!)  u=undo  s=solve  w=write  q=quit')

    import time
    t_start, t_warn = time.time(), 0.0
    grid_rot = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.img_msg is None or node.K is None:
            now = time.time()
            if now - t_start > 5 and now - t_warn > 5:
                t_warn = now
                node.get_logger().warn(
                    f'No data yet: image {"OK" if node.img_msg is not None else "MISSING"} '
                    f'({args.image}), camera_info {"OK" if node.K is not None else "MISSING"} '
                    f'({args.info}). Is realsense2_camera running? Check: ros2 topic list | grep color')
            continue
        img = node.bridge.imgmsg_to_cv2(node.img_msg, 'bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if args.target == 'charuco':
            ch_c, ch_i, m_c, m_i = detect_charuco(gray, board, d)
            pose = charuco_pose(ch_c, ch_i, board, node.K, node.D)
        else:
            pose, m_c, m_i = grid_pose(gray, d, grid_pts, node.K, node.D, rotation=grid_rot)
            if pose and grid_rot is None and pose[2] < 1.0:
                grid_rot = pose[4]   # lock the print orientation once a clean fit is found
                print(f'Tag orientation locked: {grid_rot * 90} deg vs OpenCV corner order.')
            if pose:
                pose = pose[:4]
        if m_i is not None and len(m_i):
            ARUCO.drawDetectedMarkers(img, m_c, m_i)
        if pose:
            rvec, tvec, err, n = pose
            draw_axes(img, node.K, node.D, rvec, tvec, axis_len)
            unit = 'corners' if args.target == 'charuco' else 'tags'
            status = f'board OK: {n} {unit}, reproj {err:.2f}px, dist {np.linalg.norm(tvec):.2f} m'
            colour = (0, 200, 0) if err < 1.0 else (0, 165, 255)
        else:
            status, colour = 'board NOT detected', (0, 0, 255)
        cv2.putText(img, f'samples: {len(samples)}', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(img, status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow('handeye', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if not pose:
                print('No board pose - not captured.')
                continue
            if pose[2] > 1.0:
                print(f'Reprojection {pose[2]:.2f}px too high (blur?) - not captured.')
                continue
            T_bt = node.lookup(args.base, args.tool)
            if T_bt is None:
                continue
            if cam_link_T is None and args.cam_link:
                cam_link_T = node.lookup(args.cam_link, node.optical_frame)
            samples.append({'T_base_tool': T_bt.tolist(),
                            'rvec': pose[0].flatten().tolist(),
                            'tvec': pose[1].flatten().tolist(),
                            'reproj_px': pose[2], 'corners': pose[3]})
            print(f'Captured sample {len(samples)} (reproj {pose[2]:.2f}px, {pose[3]} '
                  f'{"corners" if args.target == "charuco" else "tags"})')
        elif key == ord('u') and samples:
            samples.pop()
            print(f'Removed last sample; {len(samples)} left.')
        elif key == ord('s'):
            solve(samples, cam_link_T, (args.tool, node.optical_frame))
        elif key in (ord('w'), ord('q')):
            with open(args.out, 'w') as f:
                json.dump({'samples': samples, 'cam_link_T': cam_link_T.tolist()
                           if cam_link_T is not None else None,
                           'tool': args.tool, 'optical': node.optical_frame,
                           'K': node.K.tolist(), 'D': node.D.tolist()}, f, indent=1)
            print(f'Wrote {len(samples)} samples to {args.out}')
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()


def parse_size(s):
    a, b = s.lower().split('x')
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--image', default='/camera/camera/color/image_raw')
    common.add_argument('--info', default='/camera/camera/color/camera_info')

    p_id = sub.add_parser('identify', parents=[common])
    p_id.add_argument('--file', help='identify from an image file instead of the live topic')
    p_id.add_argument('--try-sizes', default='6x8,8x6',
                      help='comma list of squares XxY to test, e.g. 6x8,8x6,5x7')

    p_cap = sub.add_parser('capture', parents=[common])
    p_cap.add_argument('--target', choices=['charuco', 'grid'], default='charuco',
                       help='charuco board, or a grid of AprilTags/ArUco markers')
    p_cap.add_argument('--dict', required=True)
    # ChArUco options
    p_cap.add_argument('--squares', type=parse_size, help='[charuco] squares across x down, e.g. 6x8')
    p_cap.add_argument('--square', type=float, help='[charuco] square side in metres')
    p_cap.add_argument('--marker', type=float, help='[charuco] marker side in metres')
    p_cap.add_argument('--legacy', action='store_true', help='[charuco] board printed with old OpenCV layout')
    # Tag-grid options
    p_cap.add_argument('--grid', type=parse_size, help='[grid] tags across x down, e.g. 5x7')
    p_cap.add_argument('--tag', type=float, help='[grid] tag side (black square) in metres')
    p_cap.add_argument('--pitch-x', type=float, help='[grid] centre-to-centre spacing across, metres')
    p_cap.add_argument('--pitch-y', type=float, help='[grid] centre-to-centre spacing down, metres')
    p_cap.add_argument('--first-id', type=int, default=0, help='[grid] id of the top-left tag')
    p_cap.add_argument('--base', default='base_link')
    p_cap.add_argument('--tool', default='tool0')
    p_cap.add_argument('--cam-link', default='camera_link',
                       help='RealSense root frame, used to print a tool0 -> camera_link transform')
    p_cap.add_argument('--out', default='handeye_samples.json')

    p_sol = sub.add_parser('solve')
    p_sol.add_argument('file')

    args = ap.parse_args()

    if args.mode == 'identify':
        args.try_sizes = [parse_size(s) for s in args.try_sizes.split(',')]
        if args.file:
            gray = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                sys.exit(f'Could not read {args.file}')
            identify_image(gray, args.try_sizes)
            return
    if args.mode == 'capture':
        need = (['squares', 'square', 'marker'] if args.target == 'charuco'
                else ['grid', 'tag', 'pitch_x', 'pitch_y'])
        missing = [n for n in need if getattr(args, n) is None]
        if missing:
            sys.exit(f'--target {args.target} needs: ' +
                     ', '.join('--' + m.replace("_", "-") for m in missing))
    if args.mode == 'solve':
        with open(args.file) as f:
            data = json.load(f)
        solve(data['samples'], data.get('cam_link_T'),
              (data.get('tool', 'tool0'), data.get('optical', 'camera_color_optical_frame')))
        return
    run_ros(args)


if __name__ == '__main__':
    main()
