# OpenCV compatibility CUDA prototype

This is a separate opt-in frontend (`gpu-opencv-compat`). The existing
fixed-threshold CUDA detector, CPU tracker and recorded campaigns are preserved.

The device performs replicated-border box-mean adaptive thresholding at all
configured window sizes, Suzuki RETR_LIST / CHAIN_APPROX_NONE border following,
closed Douglas-Peucker approximation, convex quad filtering, nearest-neighbor
candidate rectification and integer corner-ROI extraction. Images stay on CUDA.
The scanner runs sequentially inside one threshold plane to preserve traversal
and tie order; different image/threshold planes run concurrently.

CPU stages retain OpenCV's stable candidate grouping / hierarchy, Otsu,
dictionary identification, homography matrices and cornerSubPix. Grouping is a
native adaptation of the reference rules with up to eight CPU workers. This is
not an all-CUDA detector. Its goal is equivalent outputs while removing full
frame downloads. Dense 61-marker pads still require appreciable corner patches.

Supported scope: classic ArUco, one dictionary, refinement NONE or SUBPIX,
configured adaptive threshold windows. ArUco3 and AprilTag-specific candidate
extraction (CORNER_REFINE_APRILTAG) are explicitly rejected. AprilTag dictionaries
are supported by classic ArUco decoding. The dictionary uses the runtime
OpenCV's own correction settings. No metric marker dimensions enter detection.
Candidate capacity overflow raises an error; detections are never truncated.

Subpixel refinement uses sparse host rasters in original global coordinates,
filled with CUDA-extracted neighborhoods of radius 3*maximum_window+4. This avoids
coordinate-translation rounding differences. Pathological iterations that leave
these neighborhoods are outside the prototype's proven scope; correctness must
be checked against the reference on the intended corpus. Do not claim universal
bitwise equivalence from a finite corpus.

Build with `aruco_landing.gpu_opencv.build_library`, set
`ARUCO_OPENCV_CUDA_LIBRARY`, and run `scripts/qualify_gpu_opencv.py`. That script
checks ordered IDs and corner coordinates, including blank/noisy frames and
varying scale/rotation/lighting. Benchmarks exclude rendering and compare to
eight CPU workers including grayscale download. Switching evaluation defaults
requires native-runtime equivalence and throughput evidence.

Pinned upstream algorithms: OpenCV 4.13.0. CPU runtime 4.14 must be separately
qualified; newer OpenCV implementations may change semantics. See the shipped
`opencv-LICENSE` and the retained approximation source notice below.

## Upstream source hashes

- [aruco_detector.cpp](https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/aruco/aruco_detector.cpp): SHA-256 `d0aa18045c4dfb4e4acf60a4d9a812b261dfa537c372db262556addac91b8b34`
- [contours_new.cpp](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/contours_new.cpp): SHA-256 `52627e190b93d15e7d0e9ca312e7e3350901e302fe5769a914260b9024919284`
- [contours_common.hpp](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/contours_common.hpp): SHA-256 `4863a1686b476ef1aac04bf0b8c580ccd15c17594f2b68f3c93aa2f63884bac3`
- [approx.cpp](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/approx.cpp): SHA-256 `d2b9fa2419c9eb6f880676e862d5b5fb38cd250b2db26da3638efc64181d4bcb`

## Retained upstream approximation notice

```text
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
```
