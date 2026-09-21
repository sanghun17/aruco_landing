#!/bin/bash
# Compatibility alias: all ArUco perception entrypoints use the same pipeline.
exec "$(dirname "$(readlink -f "$0")")/run.sh" "$@"
