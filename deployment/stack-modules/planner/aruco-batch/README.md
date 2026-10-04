# Batch landing policy

The pure Python package reuses `PhysicalPadDetector.estimate` (IPPE + LM and
outlier rejection), `yaw_feedback`, and shared horizontal PD/filter equations.
The existing ROS controller calls those same numerical helpers. ROS transport
and wall-clock timers are absent from the batch path.

`LandingPolicy` receives only timestamped optical pad-relative poses. Simulation
timestamps govern delay, derivative dt, marker loss and touchdown prediction.
Initial samples use `(seed, trial_id)` and are independent of batching order.

The optional `gpu-experimental` detector uses CUDA connected components,
quad/edge fitting and DICT_4X4_100 decoding. It transfers only counts, IDs and
corners. It assumes dark ink/bright paper with a fixed threshold; it does not
replace OpenCV's adaptive detector by default. Run `scripts/benchmark_gpu_aruco.py`
to compile/test it. Synthetic speedups exclude rendering and CPU PnP.

Clone the source at this module's locked owner revision before running. The stack
owns cameras, pad selection, physics adapters, experiment settings and results.

## CUDA dictionaries

The experimental detector supports DICT_4X4_100 and DICT_6X6_50. The 6x6 path
uses a separate C ABI entry point and a 64-bit code table. Existing 4x4 binaries
remain usable for 4x4 evaluation; rebuild to use 6x6. Dictionary support does not
establish accuracy equivalence to the CPU OpenCV reference detector.
