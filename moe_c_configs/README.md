# MoE C Config Layout

Marlin configs are grouped first by GPU architecture and then by quantization dtype:

- gfx936/int8_w8a8/
- gfx92a/int8_w8a8/
- gfx92a/int8_w8a16/
- gfx92a/int8_w4a8/
- gfx92a/int4_w4a16/
- gfx928/int8_w8a8/
- gfx928/int8_w8a16/
- gfx928/int8_w4a8/
- gfx928/int4_w4a16/
- gfx938/int8_w8a8/
- gfx938/fp8_w8a8/
- gfx936/w16a16/
- gfx938/w16a16/

Inside those folders, filenames do not include gfx_version or num_cus; the
architecture is selected by the directory and CU count is intentionally ignored.

Configs whose filenames include block_shape stay in this directory because
they are CUDA/block-wise specializations. The Python config loader first checks
the architecture/dtype directory for Marlin configs, then falls back to the old
dtype directory and root directory for compatibility.
