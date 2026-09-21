# ArUco perception

There is one runtime pipeline and one canonical entrypoint: `run.sh`.

```bash
bash scripts/perception_aruco-landing.sh
# Equivalent module entrypoint:
./setup.sh run aruco-landing-jetson perception/aruco-landing
```

Both run `/physical_pad_estimator` through `physical_pad_estimator.launch`.
The historical `run_detector.sh`, `run_estimator.sh` and
`run_physical_estimator.sh` filenames are compatibility aliases to `run.sh`;
they no longer select different algorithms.

The selected estimator uses all accepted marker corners for a joint planar
PnP fit, including both planar hypotheses and reprojection outlier checks.
It then applies the measured camera/body transform, learns the global-pad
alignment online for this session, and publishes pad-body/global-body poses
and subscriber-driven annotated images. The image display rate is independent
of the 60 Hz inference rate limit.

The earlier research implementation in `paper_pad_estimator_node.cpp` computes
individual marker poses and fuses them with RANSAC. That source/launch remains
available to reproduce historical experiments, but it is not selected by any
normal module entrypoint. RANSAC fusion and joint-corner PnP are different pose
estimators; calibration and online coordinate alignment do not require the
RANSAC method. The user selected joint-corner PnP for the unified runtime.

Camera capture, OptiTrack, landing control, MAVROS and flight-safety retain
independent lifecycles. See the stack's
`stacks/aruco-landing-jetson/docs/physical_pad_estimator.md` for pose topics,
frame conventions, placement reset and detected-image viewing.
