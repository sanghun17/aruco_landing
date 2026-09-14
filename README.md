# aruco_landing

ROS Noetic perception and planning components for autonomous ArUco landing.
The same core package is intended to run in simulation and on the Jetson; only
camera, odometry, and vehicle-control adapters differ.

Current scope:

- standard ROS messages only
- arbitrary YAML marker layouts, including the 61-marker Yang baseline
- per-marker subpixel detection + IPPE/LM PnP
- score-weighted translation/yaw RANSAC and fused camera-frame pad pose
- calibrated-TF conversion to body pose in the pad frame
- marker-only horizontal PD / constant-rate descent command generation

The detector continues to publish 2-D detections when the camera is
uncalibrated. Metric marker pose is deliberately withheld until a valid camera
matrix is supplied.

## Topics

Inputs:

- `/landing/camera/image_raw`
- `/landing/camera/camera_info`

Outputs:

- `/landing/markers/ids` (`std_msgs/Int32MultiArray`)
- `/landing/markers/poses_camera` (`geometry_msgs/PoseArray`, calibrated only)
- `/landing/target_pose_camera` (`geometry_msgs/PoseWithCovarianceStamped`, calibrated only)
- `/landing/target_visible` (`std_msgs/Bool`, one valid-pose result per image frame)
- `/landing/vehicle_pose_pad` (`geometry_msgs/PoseWithCovarianceStamped`, TF available only)
- `/landing/camera_pose_pad` (`geometry_msgs/PoseWithCovarianceStamped`, camera origin in pad frame)
- `/landing/estimator/inlier_ids` (`std_msgs/Int32MultiArray`)
- `/landing/estimator/processing_ms` (`std_msgs/Float32`)
- `/landing/debug/image` (`sensor_msgs/Image`)

Run the paper-pad estimator:

```bash
roslaunch aruco_landing paper_pad_estimator.launch pad_size_m:=0.7
```

The publication does not specify its detection-score formula or numerical
RANSAC thresholds. This implementation defines the score as marker pixel-side
length divided by reprojection RMSE and exposes all thresholds in
`config/paper_pad_estimator.yaml`. It uses only standard ROS messages.

The real camera stays in its calibrated 1280x720 acquisition mode. The
estimator processes the centered 720x720 ROI and shifts the principal point by
the crop offset before PnP. Native 720x720 simulation images use zero offset.

## Landing controller

`landing_controller.launch` uses `/landing/vehicle_pose_pad` for horizontal
feedback and the configured vehicle/camera pose for the touchdown-height test.
After the first valid estimate, it begins horizontal PD feedback and constant-rate
descent at the same time, saturates horizontal velocity, holds fixed yaw, and
aborts after the marker-loss timeout. It publishes standard
messages under `/landing/cmd_vel_pad` and `/landing/controller/*`.

Touchdown height is independently parameterized. The current experiment uses
`landing_altitude_min_m: 0.20` and `touchdown_height_reference: camera`, so the
controller stops when the estimated camera-to-pad height reaches 0.20 m. Body
height remains available by selecting `vehicle` instead.

`Kp`, `Kd`, and marker-loss timeout remain `X` in
the manuscript. `config/common_experiment.yaml` therefore labels its current
values provisional; freeze and report them before collecting final results.

## Scalable paper/Unreal pad

`L` is the exact outside side length. One invocation emits a vector print PDF,
SVG, nearest-neighbour PNG texture, metric marker manifest, and an Unreal-scale
OBJ/MTL pair:

```bash
rosrun aruco_landing generate_paper_pad.py \
  --size-m 0.7 --output-dir ./generated-pad --prefix baseline1
```

Print the PDF at 100% / actual size; disable fit-to-page. The PDF media box is
exactly `L x L`, and the manifest records each marker's metric corners relative
to the pad center. The layout is reconstructed from Fig. 1 of the supplied
paper: `DICT_4X4_100`, IDs 1--57 and 91--94, with the four large markers exactly
four times the small-marker side length.

The core estimator/controller remains independent of AirSim and MAVROS APIs.

## OptiTrack-to-marker vision-pose adapter

`landing_vision_pose_adapter.launch` leaves MAVROS and PX4 unchanged. It is the
single producer of `/mavros/vision_pose/pose` and starts in `optitrack` mode.
While OptiTrack remains the selected source, simultaneous
`/vrpn_client_node/pure/pose` (`T_global_body`) and
`/landing/vehicle_pose_pad` (`T_pad_body`) samples estimate
`T_global_pad = T_global_body * inverse(T_pad_body)`. The globally aligned
marker result is published on `/landing/vision_pose_marker` for shadow
comparison, without feeding it to EKF2.

The hardware-safe default is `allow_marker_switch:=false`. Once a surveyed or
bag-derived pad alignment has been validated in SITL, explicitly enable the
gate. A marker switch is still rejected unless both sources are fresh, the
alignment quality passes, and the predicted pose discontinuity stays within
configured translation and rotation limits:

```bash
roslaunch aruco_landing landing_vision_pose_adapter.launch \
  allow_marker_switch:=true alignment_file:=/work/experiments/alignment.yaml
rosservice call /landing/pose_transition/select_marker "data: true"
```

Returning to OptiTrack uses `data: false`. The adapter never silently falls
back after a marker-source dropout; it stops publishing so PX4's external-
vision timeout remains observable. Source, readiness, and quality are exposed
on `/landing/vision_pose_source`, `/landing/pose_transition/ready`, and
`/landing/pose_transition/status`.

Extract a fixed pad pose after an OptiTrack-only landing flight with:

```bash
rosrun aruco_landing estimate_pad_global_pose.py flight.bag \
  --output /work/experiments/aruco-landing/pad-alignments/flight.yaml
```

Use only one publisher for `/mavros/vision_pose/pose`. In particular, do not
run the legacy flight-safety `vision_pose_mux` alongside this adapter.
