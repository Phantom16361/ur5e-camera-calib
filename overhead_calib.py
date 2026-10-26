#!/usr/bin/env python3
"""
overhead_calib.py - C2: locate the fixed overhead camera in the robot frame.

Uses the calibrated WRIST camera (hand-eye TF must be running) to measure where each
ArUco marker sits in base_link, then solves the OVERHEAD camera's pose from the same
markers seen in its image. Markers are only needed during this calibration.

Before running (all in their own terminals):
  - UR driver                 (TF base_link -> tool0)
  - RealSense                 (/camera/camera/color/...; add align_depth.enable:=true
                               to get the optional depth cross-check of the marker size)
  - hand-eye static TF        (tool0 -> camera_link)
  - overhead camera           (/overhead/image_raw + calibrated camera_info)

Run:
  python3 overhead_calib.py --dict DICT_ARUCO_ORIGINAL --size 0.056 --station bench1
  python3 overhead_calib.py ... --resume          # reload captures after a crash

Two windows open (wrist, overhead). Keys (click a window first):
  c = capture the marker(s) the WRIST camera sees now (robot stopped!)
      -> freedrive the camera ~30-40 cm above each marker, let go, wait 2 s, press c.
         Capture each marker 2-3 times from slightly different spots; they are averaged.
         Captures are saved immediately (overhead_samples_<station>.json).
  u = undo the last capture
  x = clear all captures
  o = solve the OVERHEAD camera pose. MOVE THE ARM OUT OF THE OVERHEAD VIEW FIRST:
      it averages the next 10 overhead frames and needs all markers fully visible.
  s = save the result to overhead_<station>.yaml
  q = quit

Output: base_link -> <overhead optical frame> as a static_transform_publisher command,
plus checks: per-marker reprojection error, and each marker's position predicted by the
overhead camera (solved WITHOUT that marker) vs. where the wrist camera measured it.
"""
import argparse
import json
import os
import time

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
N_OVERHEAD_FRAMES = 10


# ----------------------------------------------------------------------------- OpenCV helpers

def get_dict(name):
    did = getattr(ARUCO, name)
    if hasattr(ARUCO, 'getPredefinedDictionary'):
        return ARUCO.getPredefinedDictionary(did)
    return ARUCO.Dictionary_get(did)


def detect(gray, dictionary):
    """Returns ({id: 4x2 corners}, set of ids seen more than once). Duplicated ids are dropped."""
    if hasattr(ARUCO, 'ArucoDetector'):
        p = ARUCO.DetectorParameters()
        p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
        corners, ids, _ = ARUCO.ArucoDetector(dictionary, p).detectMarkers(gray)
    else:
        p = ARUCO.DetectorParameters_create()
        p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
        corners, ids, _ = ARUCO.detectMarkers(gray, dictionary, parameters=p)
    if ids is None:
        return {}, set()
    out, dupes = {}, set()
    for c, i in zip(corners, ids.flatten()):
        i = int(i)
        if i in out:
            dupes.add(i)
        out[i] = np.asarray(c, np.float64).reshape(4, 2)
    for i in dupes:
        out.pop(i, None)
    return out, dupes


def auto_dict(gray, wanted_ids):
    best = None
    for name in DICT_NAMES:
        found, _ = detect(gray, get_dict(name))
        good = len([i for i in found if i in wanted_ids])
        bad = len(found) - good
        score = (good, -bad, -DICT_NAMES.index(name))
        if good and (best is None or score > best[0]):
            best = (score, name)
    return best[1] if best else None


def marker_obj(size):
    """Marker corners in the marker frame, OpenCV detection order TL, TR, BR, BL."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def single_marker_pose(img_pts, size, K, D):
    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE') else cv2.SOLVEPNP_ITERATIVE
    obj = marker_obj(size)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, D, flags=flag)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    return T, err


# ----------------------------------------------------------------------------- math helpers

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


def tf_to_mat(t):
    q, p = t.transform.rotation, t.transform.translation
    M = np.eye(4)
    M[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    M[:3, 3] = [p.x, p.y, p.z]
    return M


def _rms(base_pts, img_pts, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(base_pts, rvec, tvec, K, D)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))


def solve_overhead(base_pts, img_pts, K, D):
    """base_pts Nx3 (base_link), img_pts Nx2 -> (T_cam<-base, rms px).
    Points are nearly coplanar (table), which has two plausible 'mirror' poses, so try
    several solvers, refine each, and keep the lowest error with all points in front."""
    base_pts = np.ascontiguousarray(base_pts, np.float64)
    img_pts = np.ascontiguousarray(img_pts, np.float64)
    cands = []
    for name in ('SOLVEPNP_ITERATIVE', 'SOLVEPNP_SQPNP', 'SOLVEPNP_EPNP', 'SOLVEPNP_IPPE'):
        flag = getattr(cv2, name, None)
        if flag is None:
            continue
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(base_pts, img_pts, K, D, flags=flag)
            cands += list(zip(rvecs, tvecs))
        except cv2.error:
            pass
    best = None
    for rvec, tvec in cands:
        rvec, tvec = np.array(rvec, np.float64).reshape(3, 1), np.array(tvec, np.float64).reshape(3, 1)
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(base_pts, img_pts, K, D, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        if np.any((R @ base_pts.T + tvec)[2] <= 0):
            continue
        e = _rms(base_pts, img_pts, rvec, tvec, K, D)
        if best is None or e < best[0]:
            best = (e, R, tvec)
    if best is None:
        return None, None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best[1], best[2].flatten()
    return T, best[0]


def project(T_cam_base, pts, K, D):
    rvec = cv2.Rodrigues(T_cam_base[:3, :3])[0]
    p, _ = cv2.projectPoints(pts, rvec, T_cam_base[:3, 3], K, D)
    return p.reshape(-1, 2)


def pixels_to_plane(img_pts, K, D, T_cam_base, z0):
    """Back-project pixels onto the base-frame plane z = z0. Returns Nx3 base points."""
    und = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    T_base_cam = np.linalg.inv(T_cam_base)
    C = T_base_cam[:3, 3]
    out = []
    for u, v in und:
        d = T_base_cam[:3, :3] @ np.array([u, v, 1.0])
        s = (z0 - C[2]) / d[2]
        out.append(C + s * d)
    return np.array(out)


def leave_one_out(markers, overhead_det, K, D):
    """For each marker: solve without it, predict where it lies on the table, compare (mm)."""
    ids = [i for i in markers if i in overhead_det]
    res = {}
    if len(ids) < 3:
        return res
    for k in ids:
        others = [i for i in ids if i != k]
        bp = np.concatenate([markers[i] for i in others])
        ip = np.concatenate([overhead_det[i] for i in others])
        T, _ = solve_overhead(bp, ip, K, D)
        if T is None:
            continue
        z0 = float(np.mean(markers[k][:, 2]))
        pred = pixels_to_plane(overhead_det[k], K, D, T, z0)
        res[k] = float(np.mean(np.linalg.norm(pred - markers[k], axis=1))) * 1000
    return res


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=float, required=True, help='marker black-square side, metres')
    ap.add_argument('--dict', default='auto', help='e.g. DICT_ARUCO_ORIGINAL, or auto')
    ap.add_argument('--ids', default='0,1,2,3', help='marker ids used for calibration')
    ap.add_argument('--station', default='default', help='name for the saved files')
    ap.add_argument('--resume', action='store_true', help='reload saved wrist captures')
    ap.add_argument('--base', default='base_link')
    ap.add_argument('--wrist-image', default='/camera/camera/color/image_raw')
    ap.add_argument('--wrist-info', default='/camera/camera/color/camera_info')
    ap.add_argument('--wrist-depth', default='/camera/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--over-image', default='/overhead/image_raw')
    ap.add_argument('--over-info', default='/overhead/camera_info')
    args = ap.parse_args()
    wanted = {int(x) for x in args.ids.split(',')}
    sample_file = f'overhead_samples_{args.station}.json'

    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import tf2_ros

    class Cal(Node):
        def __init__(self):
            super().__init__('overhead_calib')
            self.bridge = CvBridge()
            self.w_img = self.o_img = self.w_depth = None
            self.o_seq = 0
            self.w_K = self.w_D = self.o_K = self.o_D = None
            self.w_frame = self.o_frame = None
            self.tfb = tf2_ros.Buffer()
            self.tfl = tf2_ros.TransformListener(self.tfb, self)
            self.create_subscription(Image, args.wrist_image, lambda m: setattr(self, 'w_img', m), 5)
            self.create_subscription(Image, args.wrist_depth, lambda m: setattr(self, 'w_depth', m), 5)
            self.create_subscription(Image, args.over_image, self.o_cb, 5)
            self.create_subscription(CameraInfo, args.wrist_info, self.w_info, 5)
            self.create_subscription(CameraInfo, args.over_info, self.o_info, 5)

        def o_cb(self, m):
            self.o_img = m
            self.o_seq += 1

        def w_info(self, m):
            if self.w_K is None:
                self.w_K, self.w_D = np.array(m.k).reshape(3, 3), np.array(m.d)
                self.w_frame = m.header.frame_id

        def o_info(self, m):
            if self.o_K is None:
                self.o_K, self.o_D = np.array(m.k).reshape(3, 3), np.array(m.d)
                self.o_frame = m.header.frame_id or 'overhead_cam_optical'
                self.o_size = (m.width, m.height)
                if np.allclose(self.o_D, 0) and abs(self.o_K[0, 0] - 1.0) < 1e-6:
                    self.get_logger().warn('Overhead camera_info looks UNCALIBRATED - check overhead.yaml is loaded.')

    # ---- capture storage: list of (id, 4x3 corners in base) in capture order
    captures, batches = [], []          # batches: number of captures added per key press
    if args.resume and os.path.exists(sample_file):
        with open(sample_file) as f:
            d = json.load(f)
        captures = [(int(c['id']), np.array(c['corners'])) for c in d['captures']]
        batches = [1] * len(captures)
        print(f'Resumed {len(captures)} captures from {sample_file}')

    def save_captures():
        with open(sample_file, 'w') as f:
            json.dump({'size': args.size, 'captures': [{'id': i, 'corners': c.tolist()} for i, c in captures]},
                      f, indent=1)

    def averaged(report=False):
        """Per marker: drop captures > 5 mm from the median position (needs >= 3), then average."""
        out = {}
        for i, c in captures:
            out.setdefault(i, []).append(c)
        avg, counts = {}, {}
        for i, v in out.items():
            v = np.array(v)
            keep = np.ones(len(v), bool)
            if len(v) >= 3:
                ctrs = v.mean(axis=1)
                keep = np.linalg.norm(ctrs - np.median(ctrs, axis=0), axis=1) < 0.005
                if keep.sum() < 2:
                    keep[:] = True
                    if report:
                        print(f'  marker {i}: captures disagree with each other - recapture it (x clears all)')
                elif report and (~keep).any():
                    print(f'  marker {i}: dropped {int((~keep).sum())} outlier capture(s) of {len(v)}')
            avg[i], counts[i] = v[keep].mean(axis=0), int(keep.sum())
        return avg, counts

    rclpy.init()
    node = Cal()
    dictionary = None if args.dict == 'auto' else get_dict(args.dict)
    result = None
    collecting, last_seq = None, -1      # overhead frame averaging for 'o'
    cv2.namedWindow('wrist', cv2.WINDOW_NORMAL)
    cv2.namedWindow('overhead', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('overhead', 960, 540)
    print(__doc__.split('Two windows')[1].split('Output:')[0])
    t0, t_warn, t_dupe = time.time(), 0, 0

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        missing = [n for n, v in (('wrist image', node.w_img), ('wrist camera_info', node.w_K),
                                  ('overhead image', node.o_img), ('overhead camera_info', node.o_K))
                   if v is None]
        if missing:
            if time.time() - t0 > 5 and time.time() - t_warn > 5:
                t_warn = time.time()
                node.get_logger().warn('Waiting for: ' + ', '.join(missing))
            continue

        w = node.bridge.imgmsg_to_cv2(node.w_img, 'bgr8')
        o = node.bridge.imgmsg_to_cv2(node.o_img, 'bgr8')
        if not getattr(node, 'size_checked', False):
            node.size_checked = True
            img_size = (o.shape[1], o.shape[0])
            print(f'Overhead image {img_size[0]}x{img_size[1]}, camera_info {node.o_size[0]}x{node.o_size[1]}'
                  + ('' if img_size == tuple(node.o_size) else '   <-- MISMATCH: intrinsics do not fit this image!'))
        wg, og = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)

        if dictionary is None:
            name = auto_dict(wg, wanted) or auto_dict(og, wanted)
            if name:
                dictionary = get_dict(name)
                print(f'Dictionary detected: {name}  (pass --dict {name} next time)')
            else:
                cv2.imshow('wrist', w); cv2.imshow('overhead', o)
                cv2.waitKey(1)
                continue

        wd_all, wdup = detect(wg, dictionary)
        od_all, odup = detect(og, dictionary)
        wd = {i: c for i, c in wd_all.items() if i in wanted}
        od = {i: c for i, c in od_all.items() if i in wanted}
        dup = (wdup | odup) & wanted
        if dup and time.time() - t_dupe > 3:
            t_dupe = time.time()
            print(f'WARNING: marker id(s) {sorted(dup)} detected twice in one image - ignored. '
                  'Something else on the table looks like that marker.')

        # ---- overhead frame averaging for the solve
        if collecting is not None and node.o_seq != last_seq:
            last_seq = node.o_seq
            collecting.append(od)
            if len(collecting) >= N_OVERHEAD_FRAMES:
                frames, collecting = collecting, None
                print()
                markers, counts = averaged(report=True)
                use = [i for i in markers if all(i in f for f in frames)]
                partly = [i for i in markers if i not in use and any(i in f for f in frames)]
                if partly:
                    print(f'Markers {partly} flickered in the overhead view (partly hidden/blurred?) - not used.')
                if len(use) < 3:
                    print(f'Need >= 3 markers captured by the wrist AND steadily visible overhead (have {use}).')
                else:
                    od_avg = {i: np.mean([f[i] for f in frames], axis=0) for i in use}
                    jitter = {i: float(np.max(np.std([f[i] for f in frames], axis=0))) for i in use}
                    bp = np.concatenate([markers[i] for i in use])
                    ip = np.concatenate([od_avg[i] for i in use])
                    T_cam_base, rms = solve_overhead(bp, ip, node.o_K, node.o_D)
                    if T_cam_base is None:
                        print('Solve failed.')
                    else:
                        T_base_cam = np.linalg.inv(T_cam_base)
                        q = R_to_quat(T_base_cam[:3, :3])
                        loo = leave_one_out({i: markers[i] for i in use}, od_avg, node.o_K, node.o_D)
                        print(f'\nOverhead solve with markers {use}: reprojection {rms:.2f}px (want < 2)')
                        print('  per marker: captures | overhead jitter px | reprojection px | height mm | '
                              'leave-one-out mm')
                        for i in use:
                            e = float(np.sqrt(np.mean(np.sum(
                                (project(T_cam_base, markers[i], node.o_K, node.o_D) - od_avg[i]) ** 2, axis=1))))
                            print(f'    marker {i}:  {counts[i]:2d}  |  {jitter[i]:5.2f}  |  {e:6.2f}  |  '
                                  f'{np.mean(markers[i][:, 2]) * 1000:6.1f}  |  {loo.get(i, float("nan")):6.1f}')
                        # --- diagnostics: is the wrist-measured layout consistent with the image at all?
                        xy = np.concatenate([markers[i][:, :2] for i in use])
                        raw = np.concatenate([od_avg[i] for i in use])
                        und = cv2.undistortPoints(raw.reshape(-1, 1, 2), node.o_K, node.o_D,
                                                  P=node.o_K).reshape(-1, 2)
                        for label, px in (('raw pixels', raw), ('undistorted', und)):
                            H, _ = cv2.findHomography(xy, px, 0)
                            hp = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), H).reshape(-1, 2)
                            he = float(np.sqrt(np.mean(np.sum((hp - px) ** 2, axis=1))))
                            print(f'  table-plane homography fit ({label}): {he:.2f}px')
                        print('  (homography low but solve high -> overhead intrinsics/resolution problem;'
                              ' both high -> wrist-measured positions are off)')
                        ctr = {i: markers[i].mean(axis=0) for i in use}
                        print('  wrist-measured distances between marker centres (check with a tape measure):')
                        for a in range(len(use)):
                            for b in range(a + 1, len(use)):
                                i, j = sorted((use[a], use[b]))
                                print(f'    {i}-{j}: {np.linalg.norm(ctr[i][:2] - ctr[j][:2]) * 1000:6.1f} mm')
                        for i in use:
                            same = np.array([cc.mean(axis=0) for ii, cc in captures if ii == i])
                            devs = np.linalg.norm(same - np.median(same, axis=0), axis=1) * 1000
                            print(f'  marker {i} captures, mm from their median: '
                                  + ', '.join(f'{d:.1f}' for d in devs))
                        if loo:
                            worst = max(loo.values())
                            verdict = 'good' if worst < 10 else ('usable' if worst < 20 else 'poor')
                            print(f'  -> worst leave-one-out {worst:.1f} mm: {verdict} (need ~10-20 mm)')
                        dist = np.linalg.norm(T_base_cam[:3, 3])
                        print(f'  overhead camera at base ({T_base_cam[0,3]:.3f}, {T_base_cam[1,3]:.3f}, '
                              f'{T_base_cam[2,3]:.3f}) m, {dist:.2f} m from the robot base')
                        cmd = ('ros2 run tf2_ros static_transform_publisher '
                               f'--x {T_base_cam[0,3]:.5f} --y {T_base_cam[1,3]:.5f} --z {T_base_cam[2,3]:.5f} '
                               f'--qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} '
                               f'--frame-id {args.base} --child-frame-id {node.o_frame}')
                        print('\nPublish it with:\n  ' + cmd + '\n')
                        result = (T_base_cam, rms, loo, cmd)
                        # draw where the solved camera thinks the markers are
                        for i in use:
                            p = project(T_cam_base, markers[i], node.o_K, node.o_D)
                            cv2.polylines(o, [p.astype(np.int32)], True, (255, 0, 255), 3)

        # ---- draw
        _, counts = averaged()
        for img, det, scale in ((w, wd, 1.0), (o, od, 2.0)):
            for i, c in det.items():
                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), int(2 * scale))
                cv2.circle(img, tuple(c[0].astype(int)), int(5 * scale), (255, 0, 0), -1)   # corner 0
                cv2.putText(img, f'{i} ({counts.get(i, 0)})', tuple(c[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 255), int(2 * scale))
        cv2.putText(w, f'wrist sees {sorted(wd)} | captured {dict(sorted(counts.items()))}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        status = f'overhead sees {sorted(od)}'
        if collecting is not None:
            status += f'  | averaging {len(collecting)}/{N_OVERHEAD_FRAMES} - keep the arm clear'
        cv2.putText(o, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 3)
        cv2.imshow('wrist', w)
        cv2.imshow('overhead', o)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if not wd:
                print('Wrist camera sees no calibration marker.')
                continue
            try:
                T_base_cam = tf_to_mat(node.tfb.lookup_transform(args.base, node.w_frame, Time()))
            except Exception as e:  # noqa: BLE001
                print(f'TF {args.base} -> {node.w_frame} unavailable (UR driver + hand-eye TF running?): {e}')
                continue
            depth = None
            if node.w_depth is not None:
                try:
                    depth = node.bridge.imgmsg_to_cv2(node.w_depth, 'passthrough').astype(np.float64)
                    if depth.shape[:2] != wg.shape[:2]:
                        depth = None
                except Exception:  # noqa: BLE001
                    depth = None
            # only the marker nearest the image centre, and only if it is reasonably central
            hh, ww = wg.shape[:2]
            i_c = min(wd, key=lambda k: np.hypot(*(wd[k].mean(axis=0) - [ww / 2, hh / 2])))
            mx, my = wd[i_c].mean(axis=0)
            if abs(mx - ww / 2) > ww * 0.25 or abs(my - hh / 2) > hh * 0.25:
                print(f'Marker {i_c} is too far from the centre of the wrist view - move the camera over it.')
                continue
            added = 0
            for i, c in ((i_c, wd[i_c]),):
                pose = single_marker_pose(c, args.size, node.w_K, node.w_D)
                if pose is None:
                    continue
                T_cam_m, err = pose
                if err > 1.5:
                    print(f'  marker {i}: reprojection {err:.2f}px too high (blur?), skipped')
                    continue
                corners_base = (T_base_cam @ T_cam_m @ np.c_[marker_obj(args.size), np.ones(4)].T).T[:, :3]
                captures.append((i, corners_base))
                added += 1
                ctr = corners_base.mean(axis=0)
                msg = (f'  marker {i}: captured at base ({ctr[0]:.4f}, {ctr[1]:.4f}, {ctr[2]:.4f}) m, '
                       f'reproj {err:.2f}px, camera distance {T_cam_m[2, 3] * 1000:.0f} mm')
                if depth is not None:
                    mask = np.zeros(wg.shape[:2], np.uint8)
                    cv2.fillConvexPoly(mask, c.astype(np.int32), 1)
                    vals = depth[(mask > 0) & (depth > 0)]
                    if vals.size > 20:
                        dz = float(np.median(vals))
                        msg += f' | depth camera says {dz:.0f} mm (ratio {T_cam_m[2, 3] * 1000 / dz:.3f})'
                print(msg)
                prev = [cc.mean(axis=0) for ii, cc in captures[:-1] if ii == i]
                if prev:
                    dev = np.linalg.norm(ctr - np.median(prev, axis=0)) * 1000
                    flag = '  <-- BAD? marker moved or blurred: press u to undo' if dev > 5 else ''
                    print(f'             {dev:.1f} mm from the earlier captures of marker {i}{flag}')
            if added:
                batches.append(added)
                save_captures()

        elif key == ord('u'):
            if batches:
                n = batches.pop()
                del captures[-n:]
                save_captures()
                print(f'Removed last capture ({n} marker(s)). Now: {averaged()[1]}')

        elif key == ord('x'):
            captures, batches = [], []
            save_captures()
            print('All captures cleared.')

        elif key == ord('o'):
            print(f'Averaging {N_OVERHEAD_FRAMES} overhead frames - keep the arm out of the overhead view...')
            collecting, last_seq = [], node.o_seq

        elif key == ord('s'):
            if result is None:
                print('Nothing to save yet; press o first.')
                continue
            T, rms, loo, cmd = result
            q = R_to_quat(T[:3, :3])
            fn = f'overhead_{args.station}.yaml'
            with open(fn, 'w') as f:
                f.write(f'# overhead camera pose, station {args.station}, {time.strftime("%Y-%m-%d %H:%M")}\n')
                f.write(f'parent: {args.base}\nchild: {node.o_frame}\n')
                f.write(f'xyz: [{T[0,3]:.5f}, {T[1,3]:.5f}, {T[2,3]:.5f}]\n')
                f.write(f'quat_xyzw: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n')
                f.write(f'reprojection_px: {rms:.3f}\n')
                f.write('leave_one_out_mm: {' + ', '.join(f'{i}: {e:.1f}' for i, e in loo.items()) + '}\n')
                f.write(f'command: "{cmd}"\n')
            print(f'Saved {fn}')

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
