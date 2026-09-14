# Experimental-validation implementation status

## Manuscript correction: estimator

The sentence saying that all inlier corners are fused in a single PnP update
does not describe this estimator. The implemented method is:

> Each detected marker independently provides a pad-pose hypothesis through
> subpixel corner refinement, IPPE PnP, and LM refinement using its known size
> and pad-frame position. Hypotheses failing the reprojection-error gate are
> rejected. Translation/yaw RANSAC then rejects geometrically inconsistent
> hypotheses, and the inlier pad poses are fused using a detection-quality
> weight to obtain one camera-frame pad pose estimate.

## Implemented

- centered 720x720 processing ROI with crop-adjusted calibrated intrinsics
- per-marker PnP, reprojection gate, geometric RANSAC, and weighted pose fusion
- one valid-pose availability result for every processed image frame
- TF-based camera-to-body conversion to vehicle pose in the pad frame
- 60 Hz PD/constant-descent controller, speed saturation, and fixed yaw
- marker-loss abort and touchdown state
- direct AirSim body-frame adapter without MAVROS
- deterministic matched random staging-position manifest
- repeated trials with CSV/JSON metrics and per-trial RGB ROS bags
- post-hoc-only AirSim ground-truth channel

## Missing inputs or unresolved validation gaps

- proposed-pad geometry and final `L`, `N`, sizes, separation, dictionary, IDs
- final `Kp`, `Kd`, `T_loss`, and `P_min`
- measured camera-to-body transform on the physical vehicle
- physical-flight OptiTrack topic/frame and a non-auto-arming MAVROS adapter
- mass/inertia/propulsion data for a matched 280 mm pawn; the package currently
  uses AirSim's default SimpleFlight pawn
- sustained 60 Hz unique RGB frames during simultaneous control and rosbagging

Current numeric placeholders are explicitly marked provisional in
`config/common_experiment.yaml` and must be frozen before paper data collection.
