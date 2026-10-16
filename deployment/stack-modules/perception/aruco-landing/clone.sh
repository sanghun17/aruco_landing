#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DST="$ROOT/ws/aruco-landing/src/aruco_landing"
REPO="${ARUCO_LANDING_REPO:-git@github.com:sanghun17/aruco_landing.git}"
BRANCH="${ARUCO_LANDING_REVISION:-2783a1b0641bfa1186e01df63ffb34cfdd98a83c}"
bash "$ROOT/scripts/lib/clone_repo.sh" "$DST" "$REPO" "$BRANCH"
