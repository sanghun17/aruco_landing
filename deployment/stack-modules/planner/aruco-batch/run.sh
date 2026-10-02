#!/bin/bash
set -euo pipefail
export PYTHONPATH="/work/ws/aruco-landing/src/aruco_landing/src${PYTHONPATH:+:$PYTHONPATH}"
exec bash /work/modules/simulation/isaac-lab/run.sh "$@"
