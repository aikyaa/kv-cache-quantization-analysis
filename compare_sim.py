import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from roofline import kv_formats, bytes_per_value
from crosscheck import m, ROWS, COLS, BYTES_PER_CYCLE

colors = {"fp16": "#2a78d6", "int8": "#eb6834", "int4": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e5e5", "lines.linewidth": 2})

runs = pd.read_csv("results/scalesim_runs.csv")
runs = runs.set_index("name")


def elem_bytes(name):
    for fname in kv_formats:
        if "_" + fname + "_" in name:
            return bytes_per_value(kv_formats[fname])
    return 2.0


rows = []
for name, r in runs.iterrows():
    M, N, K = r["M"], r["N"], r["K"]
    eb = elem_bytes(name)
    # my model: filter (weights or kv) read once, ifmap + ofmap in fp16
    my_bytes = K * N * eb + 2 * M * K + 2 * M * N
    roof = max(M * N * K / (ROWS * COLS), my_bytes / (r["bw"] * eb))
    # weight stationary: every fold loads weights, streams M rows, drains (fit to scalesim)
    folds = math.ceil(K / ROWS) * math.ceil(N / COLS)
    fold_model = folds * (2 * ROWS + COLS + M - 2)
    sim_bytes = r["dram_filter"] * eb + 2 * r["dram_ifmap"] + 2 * r["dram_ofmap"]
    rows.append(dict(gemm=name, M=M, N=N, K=K, sim_cycles=r["cycles"], roofline_cycles=round(roof),
                     fold_model_cycles=fold_model,
                     sim_over_roofline=r["cycles"] / roof,
                     my_MB=my_bytes / 1e6, sim_MB=sim_bytes / 1e6,
                     filter_bytes_ratio=r["dram_filter"] / (K * N),
                     ofmap_ratio=r["dram_ofmap"] / (M * N)))
per_gemm = pd.DataFrame(rows)
per_gemm.to_csv("results/scalesim_compare.csv", index=False, float_format="%.4g")
print(per_gemm.to_string())


# whole layer per decode step: linear gemms once, attention gemms once per (seq, kv head)
step_rows = []
for b in [1, 16, 64]:
    for ctx in [512, 2048, 8192]:
        for fname in kv_formats:
            sim = 0
            roof = 0
            for g in ["qkv", "o", "upgate", "down"]:
                row = per_gemm[per_gemm.gemm == "%s_b%d" % (g, b)].iloc[0]
                sim += row.sim_cycles
                roof += row.roofline_cycles
            for g in ["qk", "pv"]:
                row = per_gemm[per_gemm.gemm == "%s_%s_L%d" % (g, fname, ctx)].iloc[0]
                sim += row.sim_cycles * b * m["kv_heads"]
                roof += row.roofline_cycles * b * m["kv_heads"]
            step_rows.append(dict(batch=b, ctx=ctx, kv=fname, sim_cycles=sim, roofline_cycles=roof))
steps = pd.DataFrame(step_rows)
for col in ["sim_cycles", "roofline_cycles"]:
    base = steps[steps.kv == "fp16"].set_index(["batch", "ctx"])[col]
    steps[col.replace("cycles", "speedup")] = [base[(b, c)] / v for b, c, v in
                                               zip(steps.batch, steps.ctx, steps[col])]
steps.to_csv("results/scalesim_layer_steps.csv", index=False, float_format="%.4g")
print(steps.to_string())


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
x = np.arange(len(per_gemm))
ax.bar(x, per_gemm.sim_over_roofline, color="#2a78d6", width=0.7)
ax.axhline(1, color="gray", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(per_gemm.gemm, rotation=90, fontsize=7)
ax.set_yscale("log")
ax.set_ylabel("scalesim cycles / roofline cycles")
ax.set_title("per gemm, Llama-3.2-1B layer, 128x128 ws array")

ax = axes[1]
for fname in ["int8", "int4"]:
    s = steps[(steps.kv == fname) & (steps.batch == 16)]
    ax.plot(s.ctx, s.roofline_speedup, "o-", color=colors[fname], label=fname + " roofline")
    ax.plot(s.ctx, s.sim_speedup, "s--", color=colors[fname], label=fname + " scalesim")
ax.set_xscale("log", base=2)
ax.set_xlabel("context length")
ax.set_ylabel("layer speedup over fp16 KV")
ax.set_title("batch 16")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/scalesim_vs_roofline.png", dpi=150)
print("done")
