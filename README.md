# KV-cache quantization for LLM decode

Question: for decode attention, how much do INT8/INT4 KV caches cut memory traffic,
and at what context length and batch size does decode go from memory bound to
compute bound?

Everything runs on a laptop CPU with free Python packages.

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
