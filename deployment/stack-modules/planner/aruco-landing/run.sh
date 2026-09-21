#!/bin/bash
# shellcheck disable=SC1090,SC1091
set -eo pipefail

if [ ! -f /.dockerenv ]; then
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/lib/select_stack.sh"
  dsd_select_stack "planner/aruco-landing" || exit $?
  __C="$DSD_CONTAINER"
  __S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  __R="$(cd "$(dirname "$__S")/../../.." && pwd)"
  source "$__R/scripts/lib/ensure_container.sh"
  docker start "$__C" >/dev/null 2>&1 || true
  __TT=$([ -t 1 ] && echo -it || echo -i)
  cleanup(){ docker exec "$__C" pkill -INT -f "[r]oslaunch aruco_landing landing_trial.launch" >/dev/null 2>&1 || true; }
  trap 'cleanup; exit 130' INT TERM HUP
  docker exec $__TT "$__C" bash "/work/${__S#$__R/}" "$@"; __rc=$?
  cleanup
  exit $__rc
fi

source /work/config/ros_env.sh
source /opt/ros/noetic/setup.bash
source /work/ws/aruco-landing/devel/setup.bash
source /work/ws/flight-safety/devel/setup.bash --extend
set -u
source /work/scripts/lib/ensure_roscore.sh

exec taskset -c "${CPUS_CONTROL:?}" roslaunch aruco_landing landing_trial.launch \
  config_root:="${ARUCO_CONFIG_ROOT:-/work/stacks/aruco-landing-jetson/config}" \
  auto_start_on_offboard:="${LANDING_TRIAL_AUTO_START:-false}" \
  dry_run:="${LANDING_TRIAL_DRY_RUN:-true}" \
  estimation_transition:="${LANDING_ALLOW_MARKER_SWITCH:-false}" \
  router_auto_switch:="${LANDING_AUTO_SWITCH:-false}" \
  router_fallback:="${LANDING_AUTO_FALLBACK_TO_OPTITRACK:-false}" \
  marker_loss_s:="${LANDING_MARKER_LOSS_TIMEOUT_S:-0.5}" \
  approach_speed_mps:="${LANDING_APPROACH_SPEED_MPS:-0.5}" \
  landing_horizontal_speed_mps:="${LANDING_HORIZONTAL_SPEED_MPS:-0.5}" \
  descent_speed_mps:="${LANDING_DESCENT_SPEED_MPS:-0.3}" \
  router_launch_prefix:="taskset -c ${CPUS_ESTIMATION:?}" "$@"
