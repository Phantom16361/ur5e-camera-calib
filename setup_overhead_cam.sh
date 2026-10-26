#!/usr/bin/env bash
# setup_overhead_cam.sh
# Locks a Microsoft LifeCam Studio (used as a fixed overhead camera) to fixed settings, so the
# image and intrinsic calibration stay repeatable. Values below were tuned for one lab bench:
# re-tune FOCUS / EXPOSURE / BRIGHTNESS / WB_TEMP for your lighting (v4l2-ctl -d <dev> --list-ctrls).
# Expects the intrinsic calibration at ~/.ros/camera_info/overhead.yaml (see README, C1).
# Run it BEFORE starting usb_cam, and again AFTER usb_cam is up (usb_cam can reset controls).
#
#   ./setup_overhead_cam.sh          # apply settings and print them
#   ./setup_overhead_cam.sh --launch # apply, start usb_cam on /overhead, re-apply once it's running
#
# If you change resolution, focus or the camera mount, the intrinsic calibration must be redone.

set -euo pipefail

# Stable device path (glob avoids typing the ® in the name)
LC=$(ls /dev/v4l/by-id/*LifeCam*-video-index0 2>/dev/null | head -n1 || true)
if [[ -z "${LC}" ]]; then
  echo "LifeCam Studio not found under /dev/v4l/by-id/. Is it plugged in?" >&2
  exit 1
fi

# ---- Locked settings ---------------------------------------------------------
WIDTH=1920
HEIGHT=1080
FPS=30.0
POWER_LINE=1          # 1 = 50 Hz (Australia)
FOCUS=7               # 0..40, sharpest on the table surface
EXPOSURE=39           # LifeCam only accepts: 5 10 20 39 78 156 312 625 1250 2500 5000 10000
BRIGHTNESS=30         # 30..255 (acts like gain on this camera)
WB_TEMP=3000          # 2500..10000 K
# -----------------------------------------------------------------------------

apply_settings () {
  # Order matters: switch to manual modes before setting the manual values.
  v4l2-ctl -d "${LC}" -c power_line_frequency=${POWER_LINE}
  v4l2-ctl -d "${LC}" -c focus_automatic_continuous=0
  v4l2-ctl -d "${LC}" -c focus_absolute=${FOCUS}
  v4l2-ctl -d "${LC}" -c auto_exposure=1
  v4l2-ctl -d "${LC}" -c exposure_time_absolute=${EXPOSURE}
  v4l2-ctl -d "${LC}" -c white_balance_automatic=0
  v4l2-ctl -d "${LC}" -c white_balance_temperature=${WB_TEMP}
  v4l2-ctl -d "${LC}" -c brightness=${BRIGHTNESS}
}

show_settings () {
  v4l2-ctl -d "${LC}" --list-ctrls \
    | grep -E "exposure|focus|white_balance|brightness|power_line"
}

echo "Using ${LC}"
apply_settings
show_settings

if [[ "${1:-}" == "--launch" ]]; then
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 not found. Source ROS first, e.g.: source /opt/ros/\$ROS_DISTRO/setup.bash" >&2
    exit 1
  fi
  if ! ros2 pkg prefix usb_cam >/dev/null 2>&1; then
    echo "usb_cam is not installed. Install it with:" >&2
    echo "  sudo apt install ros-\${ROS_DISTRO}-usb-cam" >&2
    exit 1
  fi
  echo "Starting usb_cam on /overhead (${WIDTH}x${HEIGHT} MJPG @ ${FPS} fps)..."
  # usb_cam can't follow the relative by-id symlink, so pass the resolved /dev/videoN.
  LC_REAL=$(readlink -f "${LC}")
  echo "Resolved device: ${LC_REAL}"
  ros2 run usb_cam usb_cam_node_exe --ros-args \
    -r __ns:=/overhead \
    -p video_device:="${LC_REAL}" \
    -p camera_name:=overhead \
    -p camera_info_url:="file://${HOME}/.ros/camera_info/overhead.yaml" \
    -p frame_id:=overhead_cam_optical \
    -p image_width:=${WIDTH} -p image_height:=${HEIGHT} \
    -p framerate:=${FPS} \
    -p pixel_format:=mjpeg2rgb &
  CAM_PID=$!
  sleep 3
  if ! kill -0 ${CAM_PID} 2>/dev/null; then
    echo "usb_cam exited during startup; see its error above." >&2
    exit 1
  fi
  echo "Re-applying settings after usb_cam start..."
  apply_settings
  show_settings
  echo "usb_cam running (PID ${CAM_PID}). Ctrl+C to stop."
  wait ${CAM_PID}
fi
