# KV-cache quantization for LLM decode

How much do INT8/INT4 KV caches cut memory traffic during decode, and at what context length
and batch size does decode stop being memory bound? Everything here runs on a laptop CPU.

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
python run_all.py          # fast parts, ~5 min
python run_all.py --all    # plus simulator and perplexity runs, a few hours
```

## Summary

- Decode is memory bound almost everywhere. For Llama-3-8B on an A100 with fp16 KV, no batch size
  is compute bound past ~760 tokens of context. INT8 KV doubles that, INT4 roughly quadruples it.
- INT4 KV speeds up a decode step 1.1x at 1k context and 3.1x at 128k (Llama-3-8B, batch 16, roofline).
- INT8 KV costs no accuracy. INT4 costs +0.1 ppl if keys are quantized per channel, +1.1 if per token.
- A weight-stationary systolic array (SCALE-Sim) gets no speedup from smaller KV at decode batch
  sizes, because array fill/drain dominates, not DRAM.
- On a CPU the gain only shows up with a fused dequant kernel, and even then the vector units cap it.
- Across chips, compressing KV makes the all-reduces a bigger share of the step. In-network
  reduction fixes most of that.

## Roofline model

`roofline.py` counts bytes and FLOPs for one decode step (batch B, context L):

- weights: 2 B/param, read once per step and shared by the batch
- KV: B x (L+1) x bytes per token
- FLOPs: 2 x params x B (linear) + 4 x layers x heads x head_dim x L x B (attention)
- vector ops: dequant of every KV value read, plus softmax

| KV format | bytes/value |
|---|---|
| fp16 | 2.0 |
| int8, g128, scale only | 1.016 |
| int4, g64, scale + zero | 0.5625 |

Hardware: A100 (312 TFLOP/s, 2.0 TB/s), H100 (989 TFLOP/s, 3.35 TB/s) and a made-up NPU
(200 TFLOP/s, 0.8 TB/s, 4 TFLOP/s vector).

Attention on its own is never compute bound. Each sequence has its own cache, so batching gives
no reuse, and the intensity is just `2 x gqa_group / bytes_per_value`: 1 FLOP/B for Llama-2-7B in fp16,
14 for Llama-3-8B in int4, against ridge points of 153 (A100) and 295 (H100).

The full step can be compute bound because weights are shared across the batch, but only at short
context. Context above which no batch is compute bound:

| model | hw | fp16 | int8 | int4 |
|---|---|---|---|---|
| Llama-2-7B | A100 | 163 | 324 | 591 |
| Llama-3-8B | A100 | 759 | 1536 | 2901 |
| Llama-3-8B | H100 | 384 | 768 | 1418 |
| Llama-3.2-1B | A100 | 497 | 1006 | 1900 |

Inside that region you still need a batch of a few hundred, and memory capacity usually runs out
first (`bound_map.png`).

INT4 KV per token is 0.28x of fp16, not 0.25x, because of the scales. The step speedup grows with
context since weights dominate at short context: 1.1x at 1k, 2x at 16k, 3.1x at 128k (`speedup.png`).

Dequant cost matters on a weak vector unit. INT4 needs ~3 vector ops per value. On the NPU at 32k
context, INT4 loses its whole speedup below ~2.5 TFLOP/s of vector compute, and below ~1.7 INT8
is faster than INT4 (`npu_vector_sweep.png`).

## SCALE-Sim cross-check

`crosscheck.py` runs every GEMM of one Llama-3.2-1B layer through SCALE-Sim (128x128 array, 1 GHz,
512 B/cycle, 32 MB SRAM) for batch 1/16/64, context 512/2k/8k and each KV format. SCALE-Sim has no
data types, so narrow KV is modeled as more words per cycle. `compare_sim.py` compares against
the roofline.

They disagree a lot:

- SCALE-Sim is 6-35x slower than the roofline. Decode GEMMs have tiny M (batch for linear layers,
  GQA group of 4 for attention), and a weight-stationary array pays a load, stream and drain for every
  128x128 weight tile regardless of M. Cycles match `folds x (2*rows + cols + M - 2)` to within one
  cycle on every GEMM that doesn't stall.
- It shows no speedup from INT4 KV (roofline: 1.9x at batch 16, 8k). The array never waits on DRAM.
- It counts traffic the roofline ignores: partial sums written out after each K fold (16-64x the
  output size), PV reading its filter twice (N=64 fills half the columns), and up/gate weights that
  don't fit in SRAM and get fetched twice.

Dataflow explains most of the gap (`dataflow.py`, cycles):

| GEMM | roofline | ws | os | is |
|---|---|---|---|---|
| qkv, batch 1 | 24,596 | 147,100 | 55,247 | 55,263 |
| qkv, batch 64 | 25,856 | 171,300 | 55,247 | 55,263 |
| down, batch 16 | 66,176 | 407,600 | 135,135 | 155,519 |
| QK, fp16, 2k | 545 | 6,175 | 5,087 | 2,429 |
| QK, int4, 2k | 180 | 6,175 | 5,087 | 2,429 |
| PV, fp16, 2k | 545 | 6,175 | 2,301 | 7,135 |

Output/input stationary gets linear layers to ~2x the roofline and attention to ~4.3x. Batch 1 and
64 cost the same, so any batch up to 128 is free on this array. INT4 still doesn't help.

## Accuracy

`quant.py` does group-wise fake quantization and real int4 packing (measures 0.5625 B/value, same
as the roofline). `ppl.py` hooks it into the model's attention. WikiText-2 test; GPT-2 on 64k tokens,
TinyLlama-1.1B (4 KV heads) on 16k.

| KV | bytes vs fp16 | GPT-2 | TinyLlama |
|---|---|---|---|
| fp16 | 1.00 | 31.17 | 7.178 |
| int8 g64 | 0.53 | 31.17 | 7.179 |
| K int8, V int4 | 0.41 | 31.23 | 7.200 |
| int4, K per-channel | 0.30 | 31.28 | 7.233 |
| int4 g32 | 0.31 | 31.98 | 7.317 |
| K int4, V int8 | 0.41 | 32.24 | 7.383 |
| int4 g64 | 0.28 | 32.32 | 7.401 |
| int2, K per-channel | 0.19 | 37.34 | 9.592 |
| int3 g32 | 0.25 | 37.96 | 7.849 |
| int2 g32 | 0.19 | 162.1 | 37.46 |
| fp8 e4m3 | 0.50 | 31.28 | - |
| fp8 e5m2 | 0.50 | 31.71 | - |

Keys are the sensitive part: int4 keys cost +1.1 ppl on GPT-2, int4 values +0.06. A few key channels
carry large outliers, so per-channel scales for keys fix most of it and take int2 from 162 to 37.
Both models give the same ordering.

FP8 moves the same bytes as int8 but is less accurate (e4m3 has 3 mantissa bits). Its advantage is
native hardware support, not traffic or accuracy (`fp8.py`).

The numbers above come from full-sequence passes. `decode_check.py` reruns GPT-2 token by token
with a real cache and gets the same perplexity to within 0.1% for per-token formats, since re-quantizing
an already quantized token gives the same values. Greedy outputs drift much faster than perplexity:
int8 matches fp16 on 76% of generated tokens, int4 g64 on 16%.

Per-channel keys can't really be quantized in decode until their group of 32 tokens is complete.
`kivi.py` does it properly (quantize full groups only, keep the newest tokens in fp16):

| KV | ppl, token by token |
|---|---|
| fp16 | 34.18 |
| int4 per-token | 34.72 |
| int4 KIVI, fp16 window 32 | 34.21 |
| int2 per-token | 129.3 |
| int2 KIVI, fp16 window 128 | 37.49 |

The fp16 window adds almost nothing at long context (128 tokens out of 32k).

## CPU check

`cpu_bench.py` runs decode attention for one Llama-3-8B layer on this laptop, dequantizing one
4096-token block at a time in PyTorch. `fused_bench.py` does the same in numba, dequantizing each
row into a small buffer that stays in L1. Best of 15 runs; the laptop was on battery and timings
moved 2-4x between runs.

| ctx | fp32 | int8 unfused | int4 unfused | int8 fused | int4 fused |
|---|---|---|---|---|---|
| 16k | 4.6 ms | 18.1 ms | 46.0 ms | 2.4 ms | 2.6 ms |
| 128k | 42.8 ms | 176.8 ms | 414.2 ms | 22.0 ms | 40.7 ms |

(fused fp32: 3.5 ms at 16k, 32.2 ms at 128k)

Unfused, smaller caches are slower: each block is converted to a new fp32 tensor in memory, so it
moves more bytes than fp32 does. Fused int8 beats fp32 by ~1.4x (another run gave 2.7x), well short
of the 3.9x byte saving, because the CPU's arithmetic becomes the limit. Unpacking int4 costs more,
so it ends up around fp32 speed at long context.

## Multi-chip

`multichip.py` models tensor parallel Llama-2-70B on H100s (NVLink 450 GB/s per direction, two
all-reduces per layer, 2 us per hop).

8 chips, batch 32, 32k context:

| setup | step | all-reduce share |
|---|---|---|
| fp16 KV, ring | 22.8 ms | 21% |
| int4 KV, ring | 13.5 ms | 35% |
| int4 KV, in-network reduction | 9.6 ms | 9% |

INT4 halves memory time but the all-reduces stay at 4.8 ms, so they matter more. Decode all-reduces
are tiny (32 x 8192 x 2 B), so they're latency bound; letting the switch do the sum cuts 2(N-1) hops
to 2 and the all-reduce time to 0.8 ms. Compressing activations to int8 on top saves another 0.1 ms.
The 2 us hop latency is a guess; sweeping 0.25-10 us changes the size of the win (1.1 to 9.7 ms) but
not the ordering.

Llama-2-70B has 8 KV heads, so past 8 chips heads get copied and per-chip KV traffic stops
shrinking. 8 to 32 chips only takes the fp16 step from 18.8 to 14.9 ms.

Moving a 32k-token cache from a prefill chip to a decode chip: 10.7 GB in fp16, 3.0 GB in int4.
That's 24 vs 6.7 ms over NVLink, 0.86 vs 0.24 s over 100 GbE.

Splitting the cache across chips by sequence would need a max-then-rescaled-sum combine, which a
sum/max switch reduction can only do in two passes.

## Limitations

- Roofline uses peak specs and perfect overlap; no paged attention, kernel launch cost or prefill.
- SCALE-Sim models a single array and fakes narrow KV with extra bandwidth.
- Perplexity is short-context WikiText-2 only; long-context quality isn't measured.
- CPU timings are noisy (battery).
- The multi-chip model is analytical with no congestion and an idealized switch reduction.

## Files

| file | |
|---|---|
| `roofline.py`, `plots.py` | roofline model and figures |
| `simulate.py`, `crosscheck.py`, `compare_sim.py`, `dataflow.py` | SCALE-Sim runs |
| `quant.py` | fake quant, KIVI keys, fp8, int4 packing |
| `ppl.py`, `plot_ppl.py`, `decode_check.py`, `kivi.py`, `fp8.py` | accuracy |
| `cpu_bench.py`, `fused_bench.py` | CPU benchmarks |
| `multichip.py` | multi-chip model |
| `run_all.py`, `tests/` | rerun everything, `python -m pytest` |

Results are in `results/` and figures in `results/figures/`.
