#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${MOE_C_BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TARGET="${MOE_C_TARGET:-moe_c_split}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DPYTHON_PATH="$(which python3)" \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
  -DTORCH_INCLUDE_PATH="$(python3 -c 'import torch;import os; print(os.path.dirname(torch.__file__))')" \
  -DPYTHON_INCLUDE_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")" \
  -DPYBIND11_INCLUDE_PATH="$(python3 -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))")" \
  -DPYTHON_LIB_PATH="$(python3 -c "import sysconfig; import os; print(sysconfig.get_config_var('LIBDIR'))")" \
  -DMOE_C_HIP_ARCHITECTURES="${MOE_C_HIP_ARCHITECTURES:-}"

cmake --build "${BUILD_DIR}" --target "${BUILD_TARGET}" -- -j"${MOE_C_BUILD_JOBS:-168}" VERBOSE=1
