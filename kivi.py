import math
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ppl import setting, kv

# KIVI style keys need a real decode loop: a key's group only gets quantized once it is full,
# so in a full-sequence pass earlier queries would see keys that weren't quantized yet at their step

N_CHUNKS = 4
LENGTH = 512
torch.set_num_threads(8)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="kvquant", dtype=torch.float32)
model.eval()
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids


def nll_decode(chunk):
    cache = DynamicCache()
    total = 0.0
    with torch.no_grad():
        for t in range(chunk.shape[1] - 1):
            out = model(chunk[:, t:t + 1], past_key_values=cache, use_cache=True)
            logp = torch.log_softmax(out.logits[0, -1], dim=-1)
            total -= logp[chunk[0, t + 1]].item()
    return total


def kivi(bits, residual, group=32):
    c = kv(bits, bits, group, group, "kivi")
    c["residual"] = residual
    return c


def stored_bytes(c, T):
    # bytes per value of the keys at the end of a window of T tokens
    if c is None:
        return 2.0
    b = c["k_bits"] / 8 + 4 / c["k_group"]
    if c["k_axis"] != "kivi":
        return b
    n = max(T - c["residual"], 0) // c["k_group"] * c["k_group"]
    return (n * b + (T - n) * 2) / T


runs = {
    "fp16": None,
    "int4 per-token": kv(4, 4, 32, 32),
    "int4 kivi, residual 0": kivi(4, 0),
    "int4 kivi, residual 32": kivi(4, 32),
    "int4 kivi, residual 128": kivi(4, 128),
    "int2 per-token": kv(2, 2, 32, 32),
    "int2 kivi, residual 0": kivi(2, 0),
    "int2 kivi, residual 32": kivi(2, 32),
    "int2 kivi, residual 128": kivi(2, 128),
}

rows = []
for name, c in runs.items():
    setting.clear()
    if c:
        setting.update(c)
    nll, n = 0.0, 0
    for i in range(N_CHUNKS):
        chunk = ids[:, i * 20000:i * 20000 + LENGTH]
        nll += nll_decode(chunk)
        n += LENGTH - 1
    rows.append(dict(kv=name, ppl_decode=math.exp(nll / n), key_bytes_per_value_at_512=stored_bytes(c, LENGTH),
                     key_bytes_per_value_at_32k=stored_bytes(c, 32768)))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv("results/kivi.csv", index=False, float_format="%.4f")
