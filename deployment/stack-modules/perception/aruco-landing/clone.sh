#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DST="$ROOT/ws/aruco-landing/src/aruco_landing"
REPO="${ARUCO_LANDING_REPO:-git@github.com:sanghun17/aruco_landing.git}"
BRANCH="${ARUCO_LANDING_REVISION:-4e23fe250a7b6996b39a11cd728b2477f9ab65cd}"
bash "$ROOT/scripts/lib/clone_repo.sh" "$DST" "$REPO" "$BRANCH"
