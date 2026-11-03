import sys
import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from roofline import models, kv_formats, hw, step_time, max_ctx_compute_bound, bytes_per_value
from ppl import setting, perplexity

# fp8 kv: 1 byte per value with no group scales, and hopper can read it natively

fp8 = dict(bits=8, group=0, meta=0, dequant=1)
formats = {"fp16": kv_formats["fp16"], "fp8": fp8, "int8": kv_formats["int8"], "int4": kv_formats["int4"]}

rows = []
m = models["llama3-8b"]
for hname in ["A100", "H100"]:
    h = hw[hname]
    for fname, f in formats.items():
        t16 = step_time(m, formats["fp16"], h, 16, 32768)[0]
        t = step_time(m, f, h, 16, 32768)[0]
        rows.append(dict(hw=hname, kv=fname, bytes_per_value=bytes_per_value(f),
                         speedup_at_32k_b16=t16 / t, max_ctx_compute_bound=round(max_ctx_compute_bound(m, f, h))))
roof = pd.DataFrame(rows)
roof.to_csv("results/fp8_roofline.csv", index=False, float_format="%.4g")
print(roof.to_string())

if "--ppl" in sys.argv:
    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="kvquant", dtype=torch.float32)
    model.eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, :65536]
    runs = {
        "fp8 e4m3": dict(fp8="e4m3", scaled=False),
        "fp8 e4m3 per-token scale": dict(fp8="e4m3", scaled=True),
        "fp8 e5m2": dict(fp8="e5m2", scaled=False),
        "fp8 e5m2 per-token scale": dict(fp8="e5m2", scaled=True),
    }
    rows = []
    for name, c in runs.items():
        setting.clear()
        setting.update(c)
        t0 = time.time()
        p = perplexity(model, ids, 1024)
        print("%-26s ppl %.3f  (%.0fs)" % (name, p, time.time() - t0), flush=True)
        rows.append(dict(model="gpt2", kv=name, ppl=p))
    pd.DataFrame(rows).to_csv("results/ppl_gpt2_fp8.csv", index=False, float_format="%.4f")
