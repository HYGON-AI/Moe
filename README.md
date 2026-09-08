# Hygon MOE_C Operators

This repository provides high-performance Mixture-of-Experts (MOE) operators for
Hygon DCU. It contains MOE GEMM kernels, common support kernels, Python dispatch
wrappers, operator configuration files, and local correctness tests.

The operator APIs and test cases are kept compatible with the AITER MOE call
path. This repository can be integrated into AITER as a third-party MOE operator
backend, and it can also be built, debugged, and validated as a standalone
`moe_c` repository.

## Supported Operators

The current source tree includes MOE GEMM kernels for the following precision
modes:

| Precision | Scale granularity | Main usage |
| --- | --- | --- |
| W4A8 | channelwise | INT4 weight + INT8 activation MOE GEMM |
| W8A8 INT8 | channelwise / blockwise / tensorwise | INT8 weight + INT8 activation MOE GEMM |
| W8A8 FP8 | channelwise / blockwise / tensorwise | FP8 weight + FP8 activation MOE GEMM |
| W4A16 | groupwise | INT4 weight + FP16/BF16 activation MOE GEMM |
| W8A16 | channelwise / blockwise | INT8 weight + FP16/BF16 activation MOE GEMM |
| WFP4A8 | channelwise / groupwise32 | FP4 weight + INT8 activation MOE GEMM |
| WFP4A16 | groupwise | FP4 weight + FP16/BF16 activation MOE GEMM |
| W16A16 | channelwise | FP16/BF16 MOE GEMM path |

The repository also includes foundational support operators required by the MOE
compute path, including top-k softmax, moe alignment, MOE sum, and activation
operators such as Silu-and-mul.

## Repository Layout

```text
.
|-- csrc/              # Torch library registration and per-precision .cu entry files
|-- csrc_for_aiter/    # Kernel implementation headers, configs, intrinsics, and utilities
|-- moe_utils/         # Python op wrappers, MOE dispatch, golden reference, shuffle, and helpers
|-- moe_c_configs/     # moe_c configs organized by GPU architecture and quantization type
|-- op_test/           # Local correctness tests migrated from the AITER MOE CI path
|-- tune/              # moe_c tuning scripts
|-- cmake/             # hipify, source copy, and CMake helper scripts
|-- CMakeLists.txt     # Standalone build configuration
`-- build.sh           # Local build entry point
```

## AITER Integration Build And Test

To validate `moe_c` through AITER, prepare the AITER build and test environment
first:

```bash
# Clone the AITER repository
git clone <aiter_repository_url>

# Enter the AITER directory
cd aiter

# Initialize and update submodules
git submodule update --init

# Install AITER
pip install -e .
```

Standard MOE operator unit tests should be run from the AITER repository root.
This triggers just-in-time compilation when needed, for example:

```bash
cd aiter

GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w4a8_perchannel.py
```

## Local Build And Test

### Build

Run the build from the repository root:

```bash
cd /home/work/workspace/moe
./build.sh
```

The default target is `moe_c_split`, which builds all split modules.

You can also use `MOE_C_TARGET` to build a single module, for example:

```bash
MOE_C_TARGET=moe_w4a8 ./build.sh
```

Common build environment variables:

```bash
# Specify GPU architecture, for example gfx936 or gfx938.
# If unset, the build uses native.
MOE_C_HIP_ARCHITECTURES=gfx938 ./build.sh

# Specify build output directory.
# The Python wrappers also load .so files from this directory.
MOE_C_BUILD_DIR=/tmp/moe_c_build ./build.sh

# Specify build parallelism.
MOE_C_BUILD_JOBS=64 ./build.sh
```

Build artifacts are generated under `build/`.

### Test

TODO: local standalone test entry points are still being consolidated. The
scripts under `op_test/` are mainly for local development and debugging.

Official MOE operator accuracy validation should use the corresponding AITER
test scripts from the AITER repository.

## Configuration Files

`moe_c_configs/` is organized by GPU architecture and quantization type, for
example:

```text
moe_c_configs/gfx938/int8_w4a8/
```

Config filenames usually include `E`, `N`, optional `K`, `dtype`, and
`is_bottom`. Runtime config selection uses `GPU_ARCHS`, quantization type, expert
count, and hidden/intermediate size.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE). Third-
party source notices are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
