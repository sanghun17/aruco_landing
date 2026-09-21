#!/bin/bash
# shellcheck disable=SC1091
set -eo pipefail
source /opt/ros/noetic/setup.bash
cd /work/ws/aruco-landing
catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null
catkin build aruco_landing
echo ">> aruco-landing workspace built"
