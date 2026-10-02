#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modules"]["planner/aruco-batch"]["revision"])' "$ROOT/config/modules.lock.json")"
bash "$ROOT/scripts/lib/clone_repo.sh" "$ROOT/ws/aruco-landing/src/aruco_landing" \
  https://github.com/sanghun17/aruco_landing.git "$revision"
