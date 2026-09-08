# Hygon MOE_C 算子仓库

本仓库提供面向 Hygon DCU 的 Mixture-of-Experts（MOE）高性能算子实现，主要包含
MOE GEMM kernel、公共支撑 kernel、Python 路由封装、算子配置和本地验证脚本。

本仓库中的算子接口和测试用例与 AITER 的 MOE 调用链保持兼容，可作为 AITER
的第三方 MOE 算子实现进行集成，也可作为独立 `moe_c` 仓库进行编译、调试和
正确性验证。

## 支持的算子

当前源码树包含以下精度模式的 MOE GEMM kernel：

| 精度模式 | scale 粒度 | 主要用途 |
| --- | --- | --- |
| W4A8 | channelwise | INT4 weight + INT8 activation MOE GEMM |
| W8A8 INT8 | channelwise / blockwise / tensorwise | INT8 weight + INT8 activation MOE GEMM |
| W8A8 FP8 | channelwise / blockwise / tensorwise | FP8 weight + FP8 activation MOE GEMM |
| W4A16 | groupwise | INT4 weight + FP16/BF16 activation MOE GEMM |
| W8A16 | channelwise / blockwise | INT8 weight + FP16/BF16 activation MOE GEMM |
| WFP4A8 | channelwise / groupwise32 | FP4 weight + INT8 activation MOE GEMM |
| WFP4A16 | groupwise | FP4 weight + FP16/BF16 activation MOE GEMM |
| W16A16 | channelwise | FP16/BF16 MOE GEMM 路径 |

此外，仓库中还包含 MOE 计算链路所需的基础支撑算子，包括 top-k softmax、moe alignment、
MOE sum、Silu-and-mul 等activation算子。

## 仓库结构

```text
.
|-- csrc/              # Torch library 注册入口和各精度物理 .cu 接口文件
|-- csrc_for_aiter/    # MOE kernel 主体实现、配置、intrinsic 和公共头文件
|-- moe_utils/         # Python 侧 op wrapper、MOE 路由、golden、shuffle 和工具函数
|-- moe_c_configs/     # 按 GPU 架构和量化类型组织的 moe_c 配置文件
|-- op_test/           # 从 AITER MOE CI 路径迁移并去 AITER 化后的本地正确性脚本
|-- tune/              # moe_c 调优脚本
|-- cmake/             # hipify、源码拷贝和 CMake 辅助脚本
|-- CMakeLists.txt     # standalone 编译配置
`-- build.sh           # 本地编译入口
```

## AITER 集成环境编译与测试

如果需要在 AITER 中验证 `moe_c`，应先准备 AITER 编译和测试环境：

```bash
# Clone AITER 仓库
git clone <aiter_repository_url>

# 进入 AITER 目录
cd aiter

# 初始化并更新 submodule
git submodule update --init

# 安装 AITER
pip install -e .
```

MOE 算子标准单元测试应在 AITER 目录下执行，触发即时编译，例如：

```bash
cd aiter

GPU_ARCHS=gfx938 python op_tests/ci_tests/test_aiter_moe_with_config_w4a8_perchannel.py
```

## 本地编译与测试

### 编译

```bash
cd /home/work/workspace/moe
./build.sh
```

默认目标是 `moe_c_split`，会编译所有 split 模块。

也可以通过 `MOE_C_TARGET` 指定编译单个模块，例如：

```bash
MOE_C_TARGET=moe_w4a8 ./build.sh
```

常用编译环境变量：

```bash
# 指定 GPU 架构，例如 gfx936、gfx938；不指定时默认使用 native。
MOE_C_HIP_ARCHITECTURES=gfx938 ./build.sh

# 指定 build 输出目录，Python wrapper 也会从该目录加载 .so。
MOE_C_BUILD_DIR=/tmp/moe_c_build ./build.sh

# 指定编译并发数。
MOE_C_BUILD_JOBS=64 ./build.sh
```

编译产物位于 `build/` 目录。

### 测试

TODO：本地 standalone 测试入口仍在整理中，`op_test/` 下的脚本主要用于本地开发
和调试。

MOE 算子正式精度验证应以 AITER 仓库中的对应测试脚本调用为准。

## 配置文件

`moe_c_configs/` 按 GPU 架构和量化类型组织，例如：

```text
moe_c_configs/gfx938/int8_w4a8/
```

配置文件命名通常包含 `E`、`N`、可选 `K`、`dtype` 和 `is_bottom` 等信息。运行时会
根据当前 `GPU_ARCHS`、量化类型、专家数、hidden/intermediate size 等信息
选择对应配置。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。第三方来源登记见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
