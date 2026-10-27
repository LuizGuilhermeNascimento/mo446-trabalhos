#!/usr/bin/env bash
set -euo pipefail
ROOT=${OPENMVS_ROOT:-$HOME/.local/openmvs}
SRC=$ROOT/src
mkdir -p "$SRC"

get() { [ -d "$SRC/$1" ] || { mkdir -p "$SRC/$1" && curl -sL "$2" | tar xz -C "$SRC/$1" --strip-components 1; }; }
get openMVS https://github.com/cdcseacave/openMVS/archive/refs/tags/v2.4.0.tar.gz
get VCG https://github.com/cdcseacave/VCG/archive/658ba36d0a5666650da6e066b4794efc5a463407.tar.gz
get nanoflann https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.9.0.tar.gz

patch() { perl -pi -e "s/\Q$2\E/$3/" "$SRC/openMVS/$1"; }
# Boost >= 1.89 no longer ships the "system" component (header-only since 1.69).
patch CMakeLists.txt "program_options system serialization" "program_options serialization"
# JPEG XL is optional in the code, but the CMake macro marks it REQUIRED.
patch libs/IO/CMakeLists.txt " REQUIRED IMPORTED_TARGET" " IMPORTED_TARGET"
# cv::IMWRITE_JPEGXL_QUALITY (= 640) only exists since OpenCV 4.11; Ubuntu ships 4.10.
patch libs/Common/Types.inl "cv::IMWRITE_JPEGXL_QUALITY" "640"
# Black textures in 2.4.0 (issue #1257), fixed upstream in eeedab7.
patch libs/Common/Types.inl "Sampler::Sample< INTERTYPE, INTERTYPE, TImage<TYPE>" "Sampler::Sample< TYPE, INTERTYPE, TImage<TYPE>"
patch libs/MVS/SceneTexture.cpp "textureSize = MAXF(textureSize, maxTextureSize);" "textureSize = MINF(textureSize, maxTextureSize);"

EXTRA=()
# OpenMVS 2.4 needs CGAL >= 6 (Ubuntu 24.04 ships 5.6); CGAL is header-only, so the source tarball is enough.
if ! grep -qsE "define CGAL_VERSION ([6-9]|[1-9][0-9])\." /usr/include/CGAL/version.h /opt/homebrew/include/CGAL/version.h; then
  [ -d "$SRC/CGAL" ] || { mkdir -p "$SRC/CGAL" && curl -sL https://github.com/CGAL/cgal/releases/download/v6.1.1/CGAL-6.1.1.tar.xz | tar xJ -C "$SRC/CGAL" --strip-components 1; }
  EXTRA+=(-DCGAL_DIR="$SRC/CGAL/lib/cmake/CGAL")
fi
if [ "$(uname)" = Darwin ]; then
  # Homebrew's opencv is 5.x and pulls Qt/VTK/FFmpeg; build a minimal OpenCV 4 instead.
  get opencv https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
  [ -f "$ROOT/lib/cmake/opencv4/OpenCVConfig.cmake" ] || {
  cmake -S "$SRC/opencv" -B "$ROOT/build/opencv" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT" \
    -DBUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,features2d,flann -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
    -DWITH_FFMPEG=OFF -DWITH_OPENEXR=OFF -DWITH_IPP=OFF
  cmake --build "$ROOT/build/opencv" -j "${JOBS:-6}"
  cmake --install "$ROOT/build/opencv"; }
  # Apple clang has no bundled OpenMP; without it every OpenMVS step runs on one core.
  OMP=$(brew --prefix libomp)
  FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  EXTRA+=(-DOpenMP_C_FLAGS="$FLAGS" -DOpenMP_CXX_FLAGS="$FLAGS" -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp
         -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib" -DOpenMVS_USE_SSE=OFF
         -DCMAKE_CXX_FLAGS="-Wno-missing-template-arg-list-after-template-kw" -DCMAKE_INSTALL_RPATH="$ROOT/lib")
fi

cmake -S "$SRC/nanoflann" -B "$ROOT/build/nanoflann" -DCMAKE_INSTALL_PREFIX="$ROOT" \
  -DNANOFLANN_BUILD_EXAMPLES=OFF -DNANOFLANN_BUILD_TESTS=OFF
cmake --install "$ROOT/build/nanoflann"

cmake -S "$SRC/openMVS" -B "$ROOT/build/openMVS" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ROOT" -DCMAKE_PREFIX_PATH="$ROOT" -DVCG_ROOT="$SRC/VCG" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_USE_BREAKPAD=OFF \
  -DOpenMVS_BUILD_VIEWER=OFF -DOpenMVS_ENABLE_TESTS=OFF -DOpenMVS_ENABLE_IPO=OFF "${EXTRA[@]}"
cmake --build "$ROOT/build/openMVS" -j "${JOBS:-6}"
cmake --install "$ROOT/build/openMVS"
ls "$ROOT/bin/OpenMVS"
