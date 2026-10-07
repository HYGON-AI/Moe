# Independent HIPC GEMM Tuning

The shared tuner supports `fp8_w8a8`, `int8_w8a8`, and `int8_w4a8`
channelwise kernels. It generates independent inputs for GEMM1 and GEMM2.
GEMM2 never calls GEMM1 or SiLU to prepare its input.

Each reference uses the unshuffled quantized weights, dequantized activations
and channel scales, and a PyTorch matrix multiplication per routed expert.
The GEMM2 reference also applies the routed topk weight. W4A8 reference scales
are distinct from the launcher scales, which contain the existing factor 1/16.

Routing, weight packing, input quantization, and reference generation run
outside the timing interval. Only the selected GEMM invocation is timed.
Every attempted candidate is saved in the benchmark CSV, with correctness,
maximum absolute difference, and failure details. Failed candidates have no
latency. Correctness cannot be skipped in this tuning path.

For each M, BM, and routing method, the tuner selects the fastest passing GEMM1 and GEMM2.
It then selects the group minimizing the sum of their latencies. This sum
is a GEMM-only objective and does not measure complete MoE latency. Pairing
requires identical BM and routing methods: both modes must use CK sorting,
or both must use moe_align. Each stage prepares metadata using its mode descriptor.

Runtime config entries store both `MODE` and `BLOCK_SIZE_M`. The runtime reads
BM directly and reuses the routing metadata across both GEMM stages.

The standalone bindings select the quantizer's kernel map and decode/prefill
family from the mode, independently of M. INT8 requires CK sorting; W4A8
requires moe_align. FP8 supports CK decode and CK/moe_align prefill. Missing
kernels and stage/layout mismatches raise errors. A full-MoE call rejects
stage pairs with different BM or routing before preparing metadata.

Local regression checks (set `MOE_C_BUILD_DIR` to the rebuilt split libraries):

```bash
python op_test/test_channelwise_config_lookup.py
HIP_VISIBLE_DEVICES=7 python op_test/check_moe_c_mode_family_dispatch.py
```

The second check covers small-M prefill and large-M decode for all three
quantizers, plus FP8 moe_align prefill, and asserts one routing call per MoE.

Example (select an idle device before running; do not use device 3):

```bash
python tune/tune_moe_hipc.py \
  --quant fp8_w8a8 --arch gfx938 --dtype bf16 \
  --e 256 --n 320 --k 7168 --topk 8 --m 1 16 1024 \
  --warmup 5 --iters 30 --device 0 \
  --bench-direct --sweep-direct --direct-inprocess \
  --out-dir tune/tune_results/independent_fp8
```

Change `--quant` to tune INT8 or W4A8. `--direct-inprocess` reuses inputs
across candidates; omit it to run candidates in separate processes.
Outputs include the candidate CSV, `best_summary.json` with each BM group's
two stage results, and shape/topk/CU-specific runtime config files. Existing
runtime configs are not needed for a sweep and are not overwritten unless
the output directory explicitly points at their directory.

The W4A8 script's `run` command delegates to this same independent tuner.
Its `bench-config` command remains an explicit full-MoE diagnostic.
