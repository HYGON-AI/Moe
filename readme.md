# Hygon AITER MOE 算子仓库

本仓库提供面向 Hygon AITER 的 Mixture-of-Experts（MOE）高性能算子实现，包含
MOE kernel 开发、调优和集成过程中使用的 C 源码组织方式。

本仓库用于承载和维护可集成到 AITER 中的 MOE kernel 源码，相关源码会按照
AITER 的集成方式进行组织、编译和调用，正确性验证和回归测试应以 AITER
仓库中的官方测试脚本为准。

## 支持的算子

当前源码树包含以下精度模式的 MOE GEMM kernel 和公共辅助 kernel：

| 精度模式 | scale 粒度 / 变体 | 主要用途 |
| --- | --- | --- |
| W4A8 | channelwise | INT4 weight + INT8 activation MOE GEMM |
| W8A8 INT8 | channelwise | INT8 weight + INT8 activation MOE GEMM |
| W8A8 INT8 | blockwise | INT8 weight + INT8 activation MOE GEMM |
| W8A8 FP8 | channelwise | FP8 weight + FP8 activation MOE GEMM |
| W8A8 FP8 | tensorwise | FP8 weight + FP8 activation MOE GEMM |
| W4A16 | groupwise | INT4 weight + FP16/BF16 activation MOE GEMM |
| W8A16 | channelwise / blockwise | INT8 weight + FP16/BF16 activation MOE GEMM |
| WFP4A8 | channelwise / groupwise | FP4 weight + INT8 activation MOE GEMM |
| WFP4A16 | groupwise | FP4 weight + FP16/BF16 activation MOE GEMM |
| W16A16 | dense fallback / reference path | FP16/BF16 MOE GEMM 路径 |

此外，仓库中还包含 MOE 常用辅助算子，包括 top-k softmax、token alignment、
MOE sum、优化版 MOE sum、SGL 风格 token alignment，以及 Silu-and-mul 激活
辅助算子。

## 仓库结构

```text
.
|-- csrc/             # 本地 Torch 算子注册和编译入口文件
|-- csrc_for_aiter/   # 面向 AITER 集成准备的共享 MOE kernel 头文件
|-- test/             # 本地开发、调优脚本和配置示例
|-- cmake/            # 本地 CMake 辅助脚本
|-- CMakeLists.txt    # standalone 本地编译配置
`-- build.sh          # 本地开发编译脚本
```

`test/` 目录主要用于本地开发和调优。开源验证、正确性测试和常规回归测试应
使用 AITER 仓库中的 `op_tests` 脚本，具体命令见下文。

## 本地编译

本地编译流程主要用于 kernel 开发。如果需要编译 standalone MOE extension，
可以在本仓库目录下执行：

```bash
mkdir -p build
./build.sh
```

常用编译环境变量：

```bash
# 显式指定 GPU 架构，例如 gfx936 或 gfx938。
MOE_C_HIP_ARCHITECTURES=gfx938 ./build.sh

# 指定编译输出目录。
MOE_C_BUILD_DIR=build ./build.sh

# 指定编译并发数。
MOE_C_BUILD_JOBS=64 ./build.sh
```

## 编译环境准备与 MOE 算子单元测试指南

### I. 安装 AITER

```bash
# 克隆 AITER 仓库
git clone git@developer.sourcefind.cn:OpenDAS/aiter.git

# 进入 AITER 目录
cd aiter

# 初始化并更新 submodule
git submodule update --init

# 以 editable 模式安装 AITER
pip install -e .
```

### II. 运行 MOE 算子单元测试脚本

所有官方 MOE 算子单元测试都应在 AITER 仓库目录下执行。以下命令统一以
`gfx938` 作为示例，其他架构可按实际环境调整 `GPU_ARCHS`。

```bash
cd aiter
```

#### 1. W4A8 Channelwise 算子单元测试

该脚本用于验证 W4A8 per-channel MOE 路径，包括 `get_aiter_moe_config`
配置选择、`aiter_moe` 调用链，以及与 W4A8 per-channel golden 结果的正确性
对比。

```bash
GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w4a8_perchannel.py
```


#### 2. W8A8 Channelwise 算子单元测试

默认运行 FP8 W8A8 channelwise 路径；如需验证 INT8 W8A8 channelwise 路径，
可显式指定 `--quant int8`。

```bash
GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w8a8_channelwise.py
```

```bash
GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w8a8_channelwise.py --quant int8
```


#### 3. W4A16 Groupwise 算子单元测试

```bash
GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w4a16.py
```


