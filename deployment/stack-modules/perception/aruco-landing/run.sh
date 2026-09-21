#!/bin/bash
# Canonical ArUco detection + pose estimation entrypoint. Camera and control run separately.
set -eo pipefail
if [ ! -f /.dockerenv ]; then
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/lib/select_stack.sh"
  dsd_select_stack "perception/aruco-landing" || exit $?
  __C="$DSD_CONTAINER"
  __S="$(readlink -f "${BASH_SOURCE[0]}")"
  __R="$(cd "$(dirname "$__S")/../../.." && pwd)"
  source "$__R/scripts/lib/ensure_container.sh"
  docker start "$__C" >/dev/null
  cleanup(){ docker exec "$__C" bash -lc 'source /opt/ros/noetic/setup.bash; source /work/config/ros_env.sh; rosnode kill /physical_pad_estimator' >/dev/null 2>&1 || true; }
  trap 'cleanup; exit 130' INT TERM HUP
  docker exec -i "$__C" bash "/work/${__S#$__R/}" "$@"
  exit $?
fi
source /opt/ros/noetic/setup.bash
source /work/config/ros_env.sh
source /work/scripts/lib/ensure_roscore.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="/work/ws/aruco-landing/src/aruco_landing/src:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="/work/ws/aruco-landing/src:${ROS_PACKAGE_PATH:-}"
# A second estimator would duplicate topic publishers and image processing.
for existing in /physical_pad_estimator /paper_pad_estimator /aruco_detector /pad_relative_vehicle_state; do
  if rosnode list 2>/dev/null | grep -Fxq "$existing"; then
    echo "ERROR: stop conflicting node $existing before the physical estimator" >&2
    exit 1
  fi
done
exec taskset -c "${CPUS_PERCEPTION:?}" roslaunch aruco_landing physical_pad_estimator.launch \
  config_root:="${ARUCO_CONFIG_ROOT:-/work/stacks/aruco-landing-jetson/config}" "$@"
