#!/bin/bash
set -euo pipefail

if [ -f /usr/include/opencv4/opencv2/aruco.hpp ]; then
  echo ">> system OpenCV already provides C++ aruco"
  exit 0
fi

version=4.5.4
prefix=/opt/opencv-aruco
tmpdir=$(mktemp -d /tmp/opencv-aruco.XXXXXX)
cleanup(){ rm -rf "$tmpdir"; }
trap cleanup EXIT

git clone --depth 1 --branch "$version" https://github.com/opencv/opencv.git "$tmpdir/opencv"
git clone --depth 1 --branch "$version" https://github.com/opencv/opencv_contrib.git "$tmpdir/opencv_contrib"

cmake -S "$tmpdir/opencv" -B "$tmpdir/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DOPENCV_EXTRA_MODULES_PATH="$tmpdir/opencv_contrib/modules" \
  -DBUILD_LIST=aruco \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_JAVA=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_GTK=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_OPENCL=OFF
cmake --build "$tmpdir/build" --parallel "$(nproc)"
cmake --install "$tmpdir/build"
echo "$prefix/lib" >/etc/ld.so.conf.d/opencv-aruco.conf
ldconfig
