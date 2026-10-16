#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DST="$ROOT/ws/aruco-landing/src/aruco_landing"
REPO="${ARUCO_LANDING_REPO:-git@github.com:sanghun17/aruco_landing.git}"
BRANCH="${ARUCO_LANDING_REVISION:-7021dba11130b2bdb1cd2a0d462891238a0e6391}"
bash "$ROOT/scripts/lib/clone_repo.sh" "$DST" "$REPO" "$BRANCH"
