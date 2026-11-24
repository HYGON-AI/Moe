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

All profiles default to `--m 1 2 4 8 16 32 64 128 256 512 1024`.
Pass `--m` explicitly for other token counts. In a direct sweep,
`--gemm1-modes` and `--gemm2-modes` restrict the candidates for either stage.

The W4A8 script's `run` command delegates to this same independent tuner.
Its `bench-config` command remains an explicit full-MoE diagnostic.

## Running inside AITER

From the AITER checkout, use `python 3rdparty/moe_c/tune/tune_moe_hipc.py`
with the same options as below. The direct sweeps use AITER's JIT bindings.
W4A16 and W8A8 use the existing `module_moe_c_kernel` module. The tuner
reads runtime configurations from `aiter/moe_c_configs` and writes results to
`--out-dir`. Use a fresh `AITER_JIT_DIR` or rebuild the modules if the cache
contains the previous W4A16 dispatcher.

## INT4 W4A16

Use the same entry point for W4A16 with FP16 or BF16 activations.
The default group size is 32. Pass `--group-size 128` to tune group-128
models. Legacy decode candidates require group size 32.

```bash
python tune/tune_moe_hipc.py --quant int4_w4a16 --bench-direct --sweep-direct \
  --e 256 --n 256 --k 2048 --topk 8 --m 1 8 128 512 1024 \
  --dtype bf16 --device 7 --out-dir tune/tune_results/independent_w4a16
```

### Tuning a new `{E, N, K, topk}` configuration in AITER

Pass the workload shape (`--e` = experts, `--n` = `inter_dim` where GEMM1 has
`2*N` columns, `--k` = `model_dim`, `--topk` = active experts; both `N` and `K`
must be multiples of 128) and the target `--m` grid. Pointing `--out-dir` at
`aiter/moe_c_configs` merges the new rows into
`aiter/moe_c_configs/tuned_moe_c_w4a16_group.csv` while preserving existing
workloads (for concurrent sweeps, use separate `--out-dir` paths and merge the
CSVs afterwards):

```bash
python 3rdparty/moe_c/tune/tune_moe_hipc.py \
  --quant int4_w4a16 --bench-direct --sweep-direct \
  --e 384 --n 512 --k 7168 --topk 8 \
  --m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --dtype bf16 --device 0 \
  --out-dir aiter/moe_c_configs
```

Repeat with `--dtype fp16` if FP16 activations are also used (each `indtype` is
keyed separately in `tuned_moe_c_w4a16_group.csv`).
Add `--group-size 128` to tune the same workload with group-128 scales.

The tuner checks the GPU's actual architecture and CU count; `--arch` and
`--cu` are optional assertions. It rejects invalid shapes using the common
mode descriptors before launching. W4A16 builds independent references from
groupwise INT4 weights and charges the separate SiLU pass to unfused GEMM1.
Unlike the channelwise profiles above, W4A16 pairs the two stage winners even
when their BMs differ. Runtime prepares both alignments when needed. Compare
`best` and `best_without_legacy_decode` in the summary, then measure full MoE
latency before drawing conclusions about removing a kernel.

The worker reuses inputs and routing across candidates for one M and records
progress in a JSONL checkpoint. After a worker fault, the parent records the
in-flight candidate and resumes. `--fresh` resets the checkpoint for the
current shape, dtype, device class and tuning options; use it after a kernel
rebuild when timings need to be refreshed. Output uses `tuned_moe_c_w4a16_group.csv`
under `--out-dir`. Give concurrent sweeps separate output directories so
their CSV writes do not overlap. The tuner uses separate checkpoints for
each group size and writes its results to distinct `q_size_k` rows in the
same CSV. It uses the shared default seed and keeps the reference tolerances in
`w4a16_tune_utils.py`. The channelwise `--max-nloop` cap does not filter W4A16
prefill candidates.

The checked-in group-32 CSV uses the generated mode table. Prefill rows
migrated from the previous mode IDs have empty `time_us` values until retuned.

For prefill, `--list-modes` reports `BM / BN / BK / NLoop`, the pipeline and
the epilogue under each shape's `prefill_tile`. BN counts physical weight
columns per N chunk, so fused SiLU writes BN/2 columns; BK is fixed at 128.
For resident A, `effective_nloop` is 1 when the shape needs a streamed
fallback. The named pipeline supplies the wave layout, LDS stages and compiler
settings. Edit the Python catalog and regenerate both HIP headers with
`python tune/gen_w4a16_mode_table.py`.

The per-shape `tile` reports `BM / BN / BK / NLoop` across quant profiles.
New W4A16 decode modes also report the epilogue and the requested and effective
B prefetch depth in that field. BN counts physical weight columns; BK is 128
per K wave. BN64 splits the reduction across two K waves;
BN128 and BN256 use one. Effective prefetch depth is capped by the number of
K steps assigned to each wave for that shape.

Both W4A16 and the channelwise profiles use the same direct sweep loop, stage
candidate selection, routing cache, fastest-stage scoring helper and CSV
writer. W4A16 reuses inputs in its resumable worker. Channelwise profiles
retain per-candidate isolation unless `--direct-inprocess` is passed. W4A16
uses the runtime align policy, charges unfused GEMM1's SiLU, and supports
different stage BMs.
