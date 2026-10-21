import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from roofline import *

colors = {"fp16": "#2a78d6", "int8": "#eb6834", "int4": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "lines.linewidth": 2,
                     "font.size": 10})
out = "results/figures/"
ctxs = np.unique(np.logspace(np.log10(128), np.log10(131072), 60).astype(int))


# 1. bytes per decode step vs context
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, name in zip(axes, ["llama2-7b", "llama3-8b"]):
    m = models[name]
    for fname, f in kv_formats.items():
        gb = [decode_step(m, f, 16, c)["bytes"] / 1e9 for c in ctxs]
        ax.plot(ctxs, gb, color=colors[fname], label=fname)
    w = 2 * num_params(m) / 1e9
    ax.axhline(w, color="gray", ls=":", lw=1)
    ax.text(ctxs[0], w * 1.1, "weights only", color="gray", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(name + ", batch 16")
    ax.set_xlabel("context length")
axes[0].set_ylabel("GB moved per decode step")
axes[0].legend()
plt.tight_layout()
plt.savefig(out + "bytes_per_step.png", dpi=150)
plt.close()


# 2. memory bound vs compute bound map
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, hname in zip(axes, ["A100", "H100", "NPU"]):
    h = hw[hname]
    m = models["llama3-8b"]
    for fname, f in kv_formats.items():
        cb = [crossover_batch(m, f, h, c) for c in ctxs]
        cap = [max_batch(m, f, h, c) for c in ctxs]
        ax.plot(ctxs, cb, color=colors[fname], label=fname + " crossover")
        ax.plot(ctxs, cap, color=colors[fname], ls="--", lw=1.2, label=fname + " memory full")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_title(hname + ", Llama-3-8B")
    ax.set_xlabel("context length")
    ax.text(150, 3, "memory bound", color="gray")
    ax.text(150, 3e4, "compute bound", color="gray")
axes[0].set_ylabel("batch size")
axes[0].legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(out + "bound_map.png", dpi=150)
plt.close()


# 3. speedup of a decode step vs fp16
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, hname in zip(axes, ["A100", "H100", "NPU"]):
    h = hw[hname]
    m = models["llama3-8b"]
    base = [step_time(m, kv_formats["fp16"], h, 16, c)[0] for c in ctxs]
    for fname in ["int8", "int4"]:
        t = [step_time(m, kv_formats[fname], h, 16, c)[0] for c in ctxs]
        ax.plot(ctxs, np.array(base) / np.array(t), color=colors[fname], label=fname)
    ax.set_xscale("log", base=2)
    ax.set_title(hname + ", Llama-3-8B, batch 16")
    ax.set_xlabel("context length")
axes[0].set_ylabel("step speedup over fp16 KV")
axes[0].legend()
plt.tight_layout()
plt.savefig(out + "speedup.png", dpi=150)
plt.close()


# 4. classic roofline with a few points
fig, ax = plt.subplots(figsize=(6, 4.5))
h = hw["A100"]
x = np.logspace(-1, 4, 200)
ax.plot(x, np.minimum(h["flops"], x * h["bw"]) / 1e12, color="black", lw=1.5)
m = models["llama3-8b"]
markers = {1: "o", 64: "s"}
for fname, f in kv_formats.items():
    for b, mk in markers.items():
        pts = []
        for c in [512, 4096, 32768]:
            s = decode_step(m, f, b, c)
            ai = s["flops"] / s["bytes"]
            pts.append((ai, min(h["flops"], ai * h["bw"]) / 1e12))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], mk + "-", color=colors[fname], ms=6, lw=1,
                label="%s, batch %d" % (fname, b))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity (flop/byte)")
ax.set_ylabel("attainable TFLOP/s")
ax.set_title("A100 roofline, Llama-3-8B decode (ctx 512 / 4k / 32k)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(out + "roofline_a100.png", dpi=150)
plt.close()


# 5. how big does the vector unit need to be for dequant to be free
fig, ax = plt.subplots(figsize=(6, 4))
m = models["llama3-8b"]
vecs = np.logspace(0, 2, 50)
for fname in ["int8", "int4"]:
    sp = []
    for v in vecs:
        h = dict(hw["NPU"])
        h["vec"] = v * 1e12
        t16 = step_time(m, kv_formats["fp16"], h, 16, 32768)[0]
        tq = step_time(m, kv_formats[fname], h, 16, 32768)[0]
        sp.append(t16 / tq)
    ax.plot(vecs, sp, color=colors[fname], label=fname)
ax.axvline(4, color="gray", ls=":", lw=1)
ax.text(4.3, 1.1, "NPU default", color="gray", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("vector unit TFLOP/s")
ax.set_ylabel("step speedup over fp16 KV")
ax.set_title("NPU, Llama-3-8B, batch 16, ctx 32k")
ax.legend()
plt.tight_layout()
plt.savefig(out + "npu_vector_sweep.png", dpi=150)
plt.close()


# tables
rows = []
for hname, h in hw.items():
    for name in ["llama2-7b", "llama3-8b", "llama3.2-1b"]:
        m = models[name]
        for fname, f in kv_formats.items():
            for c in [1024, 4096, 16384]:
                for b in [1, 16, 64]:
                    s = decode_step(m, f, b, c)
                    t, bound, tc, tv, tm = step_time(m, f, h, b, c)
                    rows.append(dict(hw=hname, model=name, kv=fname, ctx=c, batch=b,
                                     GB=s["bytes"] / 1e9, kv_share=s["kv"] / s["bytes"],
                                     ms=t * 1e3, bound=bound, fits=b <= max_batch(m, f, h, c)))
pd.DataFrame(rows).to_csv("results/decode_steps.csv", index=False, float_format="%.4g")

rows = []
for hname, h in hw.items():
    for name in ["llama2-7b", "llama3-8b", "llama3.2-1b"]:
        for fname, f in kv_formats.items():
            m = models[name]
            rows.append(dict(hw=hname, model=name, kv=fname,
                             max_ctx_compute_bound=round(max_ctx_compute_bound(m, f, h)),
                             crossover_batch_at_512=round(crossover_batch(m, f, h, 512), 1),
                             max_batch_at_512=max_batch(m, f, h, 512)))
pd.DataFrame(rows).to_csv("results/crossover.csv", index=False)
print("done")
