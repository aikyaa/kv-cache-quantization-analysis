import sys
import time
import math
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from quant import quantize_kv

# current kv setting, changed between runs
setting = {}


def kvquant_attention(module, query, key, value, attention_mask, **kwargs):
    if setting:
        key = quantize_kv(key, setting["k_bits"], setting["k_group"], setting["k_axis"])
        value = quantize_kv(value, setting["v_bits"], setting["v_group"], "token")
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("kvquant", kvquant_attention)


def kv(k_bits, v_bits, k_group=64, v_group=64, k_axis="token"):
    return dict(k_bits=k_bits, v_bits=v_bits, k_group=k_group, v_group=v_group, k_axis=k_axis)


configs = {
    "fp32 (no quant)": None,
    "fp16": kv(16, 16),
    "int8 g64": kv(8, 8),
    "int4 g64": kv(4, 4),
    "int4 g32": kv(4, 4, 32, 32),
    "int4 K per-channel": kv(4, 4, 32, 64, "channel"),
    "int3 g32": kv(3, 3, 32, 32),
    "int2 g32": kv(2, 2, 32, 32),
    "int2 K per-channel": kv(2, 2, 32, 32, "channel"),
    "K int8, V int4": kv(8, 4),
    "K int4, V int8": kv(4, 8),
}


def perplexity(model, ids, window):
    nll = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, ids.shape[1] - 1, window):
            chunk = ids[:, i:i + window]
            if chunk.shape[1] < 2:
                break
            out = model(chunk, labels=chunk)
            n = chunk.shape[1] - 1
            nll += out.loss.item() * n
            count += n
    return math.exp(nll / count)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.set_num_threads(16)

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="kvquant", dtype=torch.float32)
    model.eval()

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids
    if max_tokens:
        ids = ids[:, :max_tokens]
    print(name, "tokens:", ids.shape[1])

    rows = []
    for cname, c in configs.items():
        setting.clear()
        if c:
            setting.update(c)
        t0 = time.time()
        p = perplexity(model, ids, window)
        print("%-20s ppl %.3f  (%.0fs)" % (cname, p, time.time() - t0), flush=True)
        rows.append(dict(model=name, kv=cname, ppl=p))

    out = "results/ppl_" + name.split("/")[-1] + ".csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
