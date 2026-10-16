#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DST="$ROOT/ws/aruco-landing/src/aruco_landing"
REPO="${ARUCO_LANDING_REPO:-git@github.com:sanghun17/aruco_landing.git}"
BRANCH="${ARUCO_LANDING_REVISION:-f579de3a76149ca844370e0b0b115878defc8a41}"
bash "$ROOT/scripts/lib/clone_repo.sh" "$DST" "$REPO" "$BRANCH"
