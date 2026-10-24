# KV-cache quantization for LLM decode

Question: for decode attention, how much do INT8/INT4 KV caches cut memory traffic,
and at what context length and batch size does decode go from memory bound to
compute bound?

Everything runs on a laptop CPU with free Python packages.

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Roofline model

`roofline.py` counts bytes and flops for one decode step, `plots.py` makes the figures
and tables in `results/`.

```
python roofline.py
python plots.py
```

What the model counts per decode step (batch B, context L):

- weights: 2 bytes x params, read once and shared by the whole batch
- KV cache: B x (L+1) x bytes per token (read the cache, write the new token)
- flops: 2 x params x B for the linear layers, 4 x layers x heads x head_dim x L x B for attention
- vector ops: dequantizing every KV value read, plus softmax

KV formats (scales and zero points are counted as traffic):

| format | bits | group | bytes/value |
|---|---|---|---|
| fp16 | 16 | - | 2.0 |
| int8 | 8 | 128, scale only | 1.016 |
| int4 | 4 | 64, scale + zero | 0.5625 |

Hardware is the public peak spec (dense fp16): A100 (312 TFLOP/s, 2.0 TB/s),
H100 (989 TFLOP/s, 3.35 TB/s), and a made up NPU (200 TFLOP/s, 0.8 TB/s, only 4 TFLOP/s of vector compute).

### Findings

**KV per token** (all layers): Llama-2-7B 512 KB in fp16 (MHA), Llama-3-8B 128 KB (GQA 4x),
Llama-3.2-1B 32 KB. INT4 cuts that to ~0.28x, not 0.25x, because of the group metadata.

**Attention alone never becomes compute bound.** Each sequence has its own cache, so batching
gives attention no data reuse. Its intensity is `2 x gqa_group / bytes_per_value`:
1 flop/byte for Llama-2 fp16, 14 for Llama-3-8B int4. The A100 ridge is 153 and the H100 ridge is 295.

**The whole step is only compute bound at short context.** Weights get reused across the batch
but KV doesn't, so as L grows the step slides back to memory bound no matter how big the batch is.
The context above which no batch size is compute bound:

| model | hw | fp16 | int8 | int4 |
|---|---|---|---|---|
| Llama-2-7B | A100 | 163 | 324 | 591 |
| Llama-3-8B | A100 | 759 | 1536 | 2901 |
| Llama-3-8B | H100 | 384 | 768 | 1418 |
| Llama-3.2-1B | A100 | 497 | 1006 | 1900 |

So KV quantization roughly doubles (int8) or quadruples (int4) the context where compute
bound is even possible. Even then you need a batch of a few hundred, and memory capacity
caps the batch first in most of that region (dashed lines in `bound_map.png`).
On H100 the ridge is higher, so decode is memory bound at even shorter contexts.

**Speedup is a long-context thing.** At batch 16 on Llama-3-8B, int4 KV gives ~1.1x at 1k
context (weights dominate) and 2x by 16k, approaching 3.1x at 128k (`speedup.png`).

**Dequant isn't free on a weak vector unit.** Int4 reads 0.56 bytes per value but costs ~3
vector ops to unpack and scale. On the NPU at 32k context, int4 loses its whole speedup once
vector throughput drops below ~2.5 TFLOP/s, and below ~1.7 TFLOP/s int8 beats int4
(`npu_vector_sweep.png`).

Figures: `results/figures/bytes_per_step.png`, `bound_map.png`, `speedup.png`,
`roofline_a100.png`, `npu_vector_sweep.png`. Tables: `results/decode_steps.csv`, `results/crossover.csv`.

## Cross-check with SCALE-Sim

`simulate.py` wraps SCALE-Sim for one GEMM, `crosscheck.py` runs every GEMM of one
Llama-3.2-1B decoder layer (batch 1/16/64, context 512/2k/8k, fp16/int8/int4 KV) and
`compare_sim.py` lines them up against the roofline.

```
python crosscheck.py   # ~30 min on 10 cores
python compare_sim.py
```

Simulated chip: 128x128 weight-stationary array at 1 GHz, 512 B/cycle off chip, 32 MB SRAM.
SCALE-Sim has no data types, so narrow KV is modeled by giving the attention GEMMs more
words per cycle (512 / bytes per value, rounded down to a multiple of 64).

### Where they disagree

**SCALE-Sim is 6-35x slower than the roofline** (`scalesim_vs_roofline.png`). Decode GEMMs have
tiny M (the batch for linear layers, the GQA group of 4 for attention), and on a weight-stationary
array every 128x128 weight tile costs a load, a stream and a drain regardless of M. The cycle count
matches `folds x (2 x rows + cols + M - 2)` to within one cycle for every GEMM that doesn't stall.
The roofline assumes peak MACs, which an array can't reach unless M is about as big as the array.

**SCALE-Sim sees no speedup from KV quantization.** With enough bandwidth the array never stalls on
DRAM, so cutting KV bytes changes nothing. Roofline says int4 is 1.9x faster at batch 16 and 8k context;
SCALE-Sim says 1.0x. On this kind of chip decode is bound by array fill and drain, not memory,
so KV compression only pays off once the array is sized or mapped for small M.

**Extra DRAM traffic the roofline doesn't count:**
- partial sums go out to DRAM after every K fold (ofmap writes 16-64x the output size)
- PV re-reads its filter twice (N = 64 fills only half the array's columns)
- the up/gate weights (2048 x 16384) don't fit in SRAM, so they're fetched twice and the array stalls

Tables: `results/scalesim_runs.csv`, `scalesim_compare.csv`, `scalesim_layer_steps.csv`.

## Accuracy cost

`quant.py` does group-wise fake quantization plus real int4 packing (two values per byte with an
fp16 scale and min per group; it measures out to exactly 0.5625 bytes per value, same as the roofline
assumes). `ppl.py` plugs it into GPT-2's attention through a custom attention function, so the
keys and values every query sees are quantized.

```
python ppl.py gpt2 1024 65536
python plot_ppl.py
```

WikiText-2 test, first 64k tokens, 1024 token windows:

| KV | bytes vs fp16 | perplexity |
|---|---|---|
| fp16 | 1.00 | 31.17 |
| int8 g64 | 0.53 | 31.17 |
| K int8, V int4 | 0.41 | 31.23 |
| int4 K per-channel, V per-token | 0.30 | 31.28 |
| int4 g32 | 0.31 | 31.98 |
| K int4, V int8 | 0.41 | 32.24 |
| int4 g64 | 0.28 | 32.32 |
| int2 K per-channel | 0.19 | 37.34 |
| int3 g32 | 0.25 | 37.96 |
| int2 g32 | 0.19 | 162.1 |

- int8 is free.
- Keys are the sensitive half: int4 keys cost +1.1 ppl, int4 values only +0.06.
- Grouping keys per channel instead of per token (the KIVI trick) gets int4 to +0.1 ppl,
  and takes int2 from 162 to 37, because a few key channels have much larger outliers than the rest.
- So int4 with per-channel keys moves 30% of the fp16 bytes for almost no accuracy loss.

## Multi-chip

`multichip.py` extends the step model to tensor parallel across N chips with two all-reduces per
layer, Llama-2-70B on H100s (NVLink 450 GB/s per direction, 2 us per hop).

```
python multichip.py
```

**Compressing KV makes the interconnect matter more.** On 8 chips at batch 32, 32k context, int4 KV
cuts memory time from 17.9 ms to 8.7 ms, but the ring all-reduces stay at 4.8 ms, so their share of
the step goes from 21% to 35% (`multichip_breakdown.png`, `multichip_link_share.png`).

**In-network reduction pulls it back.** If the switch does the sum (NVLink SHARP style) each chip
sends its partial once and the latency is two hops instead of 2(N-1). All-reduce time drops from
4.8 ms to 0.8 ms and the step goes from 13.5 ms to 9.6 ms. Decode all-reduces are tiny (32 x 8192 x 2 B),
so this is mostly a latency win; compressing activations to int8 on top only saves another 0.1 ms.

**KV heads run out.** Llama-2-70B has 8 KV heads, so past 8 chips each KV head gets copied and
per-chip KV traffic stops shrinking. Going from 8 to 32 chips only speeds up a 32k step from
18.8 to 14.9 ms with fp16 KV (`multichip_scaling.csv`); int4 KV helps more here than adding chips does.

**KV migration** (prefill on one chip, decode on another): a 32k context of Llama-2-70B is 10.7 GB in
fp16 and 3.0 GB in int4, i.e. 24 ms vs 6.7 ms over NVLink or 0.86 s vs 0.24 s over 100 GbE
(`kv_migration.csv`). Compression lowers the link traffic just like it lowers HBM traffic.

Context-parallel decode (splitting the cache along the sequence) would need a softmax-aware
combine (max, then rescaled sum), which plain sum/max in-switch reduction can only do in two passes.
