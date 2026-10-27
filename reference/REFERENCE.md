# Reference values (DuploBot team, MTRN4231 lab, 26 Oct 2026)

These are **our** results, shared for comparison and sanity checks. Your own camera unit, bracket and station
will differ, so calibrate your own; use these to check that yours are in the same ballpark.

## Overhead camera: Microsoft LifeCam Studio

### Locked settings (`v4l2-ctl`; applied by `setup_overhead_cam.sh`)

| Control | Value | Default | Notes |
|---|---|---|---|
| Resolution / format | 1920×1080 MJPG @ 30 fps | | Intrinsics only valid at this resolution |
| `power_line_frequency` | 1 (50 Hz) | 2 (60 Hz) | Australia mains: stops flicker banding |
| `focus_automatic_continuous` | 0 (off) | 1 | Autofocus off, or the calibration drifts |
| `focus_absolute` | 7 | 0 | Range 0–40; sharpest on our table surface |
| `auto_exposure` | 1 (Manual Mode) | 3 (Aperture Priority) | |
| `exposure_time_absolute` | 39 | 156 | Only 5, 10, 20, 39, 78, 156, 312… work; other values give a white image |
| `white_balance_automatic` | 0 (off) | 1 | Stable colours for colour detection |
| `white_balance_temperature` | 3000 K | 4500 K | Range 2500–10000 |
| `brightness` | 30 | 133 | Range 30–255; acts like gain on this camera |

Set these **before and after** starting `usb_cam`, which can reset them.


### Applying the settings by hand (copy and run)

Plug in the LifeCam, then run these in a terminal. They work without ROS.

```bash
# 1. Find the camera (stable path that survives replugging)
LC=$(readlink -f "$(ls /dev/v4l/by-id/*LifeCam*-video-index0 | head -n1)")
echo "LifeCam is $LC"

# 2. Apply the settings. Order matters: switch auto modes OFF before setting manual values
v4l2-ctl -d "$LC" -c power_line_frequency=1          # 50 Hz mains (use 2 for 60 Hz countries)
v4l2-ctl -d "$LC" -c focus_automatic_continuous=0    # autofocus off
v4l2-ctl -d "$LC" -c focus_absolute=7                # 0-40
v4l2-ctl -d "$LC" -c auto_exposure=1                 # 1 = manual
v4l2-ctl -d "$LC" -c exposure_time_absolute=39       # ONLY 5,10,20,39,78,156,312... work
v4l2-ctl -d "$LC" -c white_balance_automatic=0       # auto white balance off
v4l2-ctl -d "$LC" -c white_balance_temperature=3000  # 2500-10000 K
v4l2-ctl -d "$LC" -c brightness=30                   # 30-255 (acts like gain)

# 3. Check they stuck
v4l2-ctl -d "$LC" --list-ctrls | grep -E "exposure|focus|white_balance|brightness|power_line"
```

**Re-run step 2 after starting `usb_cam`**: it can reset the controls when it opens the camera.
(`setup_overhead_cam.sh --launch` does all of this for you, including the re-apply.)

**Tuning for your own lighting:** with the camera streaming (`ros2 run rqt_image_view rqt_image_view`
or `cheese`), change one value at a time and watch the image:
- Too dark or too bright: step the exposure along the ladder (20 ↔ 39 ↔ 78), then fine-tune with `brightness`.
- Colours too blue or orange: change `white_balance_temperature` in steps of about 500 K.
- Blurry: try `focus_absolute` values around 0–15 with the table in view.

Once you've changed focus or resolution, redo the intrinsic calibration (C1).

**Undo everything** (back to auto): `v4l2-ctl -d "$LC" -c auto_exposure=3 -c white_balance_automatic=1 -c focus_automatic_continuous=1`

### Intrinsics (C1), 1920×1080

ROS `camera_calibration`, 13×9-square checkerboard (12×8 inner corners), 30 mm squares,
plumb_bob model. Validated at **0.39 px** mean reprojection on new board positions (`check_overhead_calib.py`).

```
fx = 1388.8   fy = 1382.2   cx = 919.3   cy = 574.4
D (k1, k2, p1, p2, k3) = [-0.0033, -0.0237, -0.0005, -0.0070, 0.0]
```
Full file: `lifecam_studio_1080p.yaml` (install as `~/.ros/camera_info/overhead.yaml`).
Identical LifeCams should be close to this, but check with `check_overhead_calib.py`; over 1 px means
calibrate that unit yourself.

## Wrist camera: Intel RealSense on the tool flange (C3)

Intrinsics: RealSense factory calibration (published on `/camera/camera/color/camera_info`).

Hand-eye: 18/18 samples, AprilTag 36h11 grid (5×7, 20 mm tags, 30.5 mm pitch), board-consistency
**2.11 mm** mean; matched an independent first pass within ~3 mm. Depends entirely on **our** bracket.

| Transform | xyz (m) | quaternion (x, y, z, w) |
|---|---|---|
| `tool0 → camera_link` (what you publish) | [-0.01615, -0.09938, 0.02493] | [-0.499904, 0.502214, -0.500659, -0.497209] |
| `tool0 → camera_color_optical_frame` (solver output) | [-0.03113, -0.09924, 0.02460] | [-0.002221, -0.002155, -0.003637, 0.999989] |

```bash
ros2 run tf2_ros static_transform_publisher --x -0.01615 --y -0.09938 --z 0.02493 \
  --qx -0.499904 --qy 0.502214 --qz -0.500659 --qw -0.497209 --frame-id tool0 --child-frame-id camera_link
```

## Overhead camera in the robot frame (C2), our bench

4 × `DICT_ARUCO_ORIGINAL` markers (56 mm), 2×2 pattern ~0.23–0.39 m apart. Reprojection 3.9 px,
leave-one-out **6.4–6.8 mm**. Camera ~1.27 m in front of the base and 0.85 m above it.
**Station- and session-specific** (the camera is on a swivel): never reuse, always re-run C2.

| Transform | xyz (m) | quaternion (x, y, z, w) |
|---|---|---|
| `base_link → overhead_cam_optical` | [1.26842, 0.06553, 0.85374] | [0.660371, 0.680313, -0.223169, -0.226450] |
