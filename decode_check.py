import math
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import ppl
from ppl import configs, setting

# does the full-sequence perplexity match a real token by token decode with the quantized cache?

N_CHUNKS = 4
LENGTH = 512
torch.set_num_threads(8)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="kvquant", dtype=torch.float32)
model.eval()
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids


def nll_full(chunk):
    with torch.no_grad():
        return model(chunk, labels=chunk).loss.item() * (chunk.shape[1] - 1)


def nll_decode(chunk):
    cache = DynamicCache()
    total = 0.0
    with torch.no_grad():
        for t in range(chunk.shape[1] - 1):
            out = model(chunk[:, t:t + 1], past_key_values=cache, use_cache=True)
            logp = torch.log_softmax(out.logits[0, -1], dim=-1)
            total -= logp[chunk[0, t + 1]].item()
    return total


def greedy(prompt, n=48):
    cache = DynamicCache()
    out_ids = []
    x = prompt
    with torch.no_grad():
        for _ in range(n):
            out = model(x, past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax()
            out_ids.append(nxt.item())
            x = nxt.view(1, 1)
    return out_ids


rows = []
ref_tokens = {}
for cname in ["fp16", "int8 g64", "int4 g64", "int4 K per-channel", "int2 g32"]:
    setting.clear()
    setting.update(configs[cname])
    full, dec, n = 0.0, 0.0, 0
    match = []
    for i in range(N_CHUNKS):
        chunk = ids[:, i * 20000:i * 20000 + LENGTH]
        full += nll_full(chunk)
        dec += nll_decode(chunk)
        n += LENGTH - 1
        g = greedy(chunk[:, :64])
        if cname == "fp16":
            ref_tokens[i] = g
        match.append(sum(a == b for a, b in zip(g, ref_tokens[i])) / len(g))
    rows.append(dict(kv=cname, ppl_full_seq=math.exp(full / n), ppl_decode=math.exp(dec / n),
                     greedy_match_vs_fp16=sum(match) / len(match)))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv("results/decode_check.csv", index=False, float_format="%.4f")
