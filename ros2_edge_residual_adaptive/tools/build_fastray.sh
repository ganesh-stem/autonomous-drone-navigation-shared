#!/bin/bash
# Builds the C++ ray casting (pybind11) into the ROS package, in Ubuntu / WSL:  bash tools/build_fastray.sh
# Needs g++, python3-dev and pybind11-dev (sudo apt install g++ python3-dev pybind11-dev). Without the .so the nodes fall back to numpy, with identical results.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../web3d_drones_edge/src/fastray_cpp/fastray.cpp"
PY=${PYTHON:-$HOME/ros2_ws/venv_edge/bin/python}
g++ -O2 -ffp-contract=off -shared -std=c++17 -fPIC -I/usr/include/pybind11 $(python3-config --includes) "$SRC" \
    -o "$HERE/../edge_residual/edge_residual_adaptive/fastray$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")"
echo built
