import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from roofline import models, kv_formats, hw, num_params, kv_bytes_per_token

colors = {"fp16": "#2a78d6", "int8": "#eb6834", "int4": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e5e5", "lines.linewidth": 2})
out = "results/figures/"

HOP_LATENCY = 2e-6  # per message, roughly what nvlink all-reduces see


def allreduce(size, n, link, innet=False, hop=HOP_LATENCY):
    # returns (bytes sent per chip, time)
    if n == 1:
        return 0, 0
    if innet:
        # switch adds up the partial sums: send once, get the result back once
        sent = size
        lat = 2 * hop
    else:
        # ring: reduce-scatter + all-gather
        sent = 2 * (n - 1) / n * size
        lat = 2 * (n - 1) * hop
    return sent, sent / link + lat


def tp_step(m, f, h, batch, ctx, n, innet=False, act_bytes=2, hop=HOP_LATENCY):
    # tensor parallel over n chips, kv heads split across chips
    P = num_params(m)
    kv_heads_per_chip = max(m["kv_heads"] / n, 1)  # can't split a kv head, so it gets copied
    kv_frac = kv_heads_per_chip / m["kv_heads"]
    hbm = 2 * P / n + batch * (ctx + 1) * kv_bytes_per_token(m, f) * kv_frac
    flops = (2 * batch * P + 4 * batch * m["layers"] * m["heads"] * m["head_dim"] * ctx) / n
    t_hbm = hbm / h["bw"]
    t_comp = flops / h["flops"]
    # two all-reduces per layer (after attention and after mlp)
    size = batch * m["d"] * act_bytes
    sent, t_one = allreduce(size, n, h["link"], innet, hop)
    t_link = 2 * m["layers"] * t_one
    return dict(t_hbm=t_hbm, t_comp=t_comp, t_link=t_link,
                time=max(t_hbm, t_comp) + t_link, link_bytes=2 * m["layers"] * sent,
                fits=hbm < 0.9 * h["mem"])


def kv_migration_time(m, f, h, ctx):
    # disaggregated serving: prefill chip ships the whole cache to a decode chip
    return ctx * kv_bytes_per_token(m, f) / h["link"]


setups = [
    ("fp16 KV, ring", "fp16", False, 2),
    ("int4 KV, ring", "int4", False, 2),
    ("int4 KV, in-network", "int4", True, 2),
    ("int4 KV, in-network, int8 acts", "int4", True, 1),
]


if __name__ == "__main__":
    m = models["llama2-70b"]
    h = hw["H100"]

    # 1. where does the time go, 8 chips
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    rows = []
    for ax, ctx in zip(axes, [4096, 32768]):
        labels = []
        hbm_t, comp_t, link_t = [], [], []
        for name, fname, innet, ab in setups:
            s = tp_step(m, kv_formats[fname], h, 32, ctx, 8, innet, ab)
            labels.append(name.replace(", ", "\n"))
            hbm_t.append(max(s["t_hbm"], s["t_comp"]) * 1e3)
            link_t.append(s["t_link"] * 1e3)
            rows.append(dict(ctx=ctx, setup=name, hbm_ms=s["t_hbm"] * 1e3, compute_ms=s["t_comp"] * 1e3,
                             link_ms=s["t_link"] * 1e3, step_ms=s["time"] * 1e3,
                             link_share=s["t_link"] / s["time"]))
        x = np.arange(len(labels))
        ax.bar(x, hbm_t, color="#2a78d6", width=0.6, label="memory / compute")
        ax.bar(x, link_t, bottom=hbm_t, color="#eb6834", width=0.6, label="all-reduce")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("Llama-2-70B on 8x H100, batch 32, ctx %dk" % (ctx // 1024))
        ax.set_ylabel("ms per decode step")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out + "multichip_breakdown.png", dpi=150)
    plt.close()
    pd.DataFrame(rows).to_csv("results/multichip_breakdown.csv", index=False, float_format="%.4g")
    print(pd.DataFrame(rows).to_string())

    # 2. link share of step time as context grows
    ctxs = np.unique(np.logspace(np.log10(256), np.log10(131072), 40).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    line_colors = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"]
    for (name, fname, innet, ab), c in zip(setups, line_colors):
        share = []
        for ctx in ctxs:
            s = tp_step(m, kv_formats[fname], h, 32, ctx, 8, innet, ab)
            share.append(s["t_link"] / s["time"])
        ax.plot(ctxs, share, color=c, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("share of step time in all-reduce")
    ax.set_title("Llama-2-70B, 8x H100, batch 32")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out + "multichip_link_share.png", dpi=150)
    plt.close()

    # 3. more chips: kv heads run out at 8 for llama2-70b
    rows = []
    for n in [1, 2, 4, 8, 16, 32]:
        for fname in kv_formats:
            s = tp_step(m, kv_formats[fname], h, 32, 32768, n, True, 2)
            rows.append(dict(chips=n, kv=fname, step_ms=s["time"] * 1e3, hbm_ms=s["t_hbm"] * 1e3,
                             link_ms=s["t_link"] * 1e3, fits=s["fits"]))
    scale = pd.DataFrame(rows)
    scale.to_csv("results/multichip_scaling.csv", index=False, float_format="%.4g")
    print(scale.to_string())

    # 4. kv cache migration between chips
    rows = []
    for ctx in [4096, 32768, 131072]:
        for fname, f in kv_formats.items():
            for mname in ["llama3-8b", "llama2-70b"]:
                rows.append(dict(model=mname, ctx=ctx, kv=fname,
                                 GB=ctx * kv_bytes_per_token(models[mname], f) / 1e9,
                                 ms_on_nvlink=kv_migration_time(models[mname], f, h, ctx) * 1e3,
                                 ms_on_100GbE=ctx * kv_bytes_per_token(models[mname], f) / 12.5e9 * 1e3))
    mig = pd.DataFrame(rows)
    mig.to_csv("results/kv_migration.csv", index=False, float_format="%.4g")
    print(mig.to_string())

    # 5. the hop latency is a guess, so sweep it
    hops = np.logspace(np.log10(0.25e-6), np.log10(10e-6), 30)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = []
    for (name, fname, innet, ab), c in zip(setups[:3], line_colors):
        t = [tp_step(m, kv_formats[fname], h, 32, 32768, 8, innet, ab, hop)["time"] * 1e3 for hop in hops]
        ax.plot(hops * 1e6, t, color=c, label=name)
        for hop, tt in zip(hops, t):
            rows.append(dict(setup=name, hop_us=hop * 1e6, step_ms=tt))
    ax.axvline(HOP_LATENCY * 1e6, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("latency per hop (us)")
    ax.set_ylabel("ms per decode step")
    ax.set_title("Llama-2-70B, 8x H100, batch 32, ctx 32k")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out + "multichip_latency_sweep.png", dpi=150)
    plt.close()
    sweep = pd.DataFrame(rows)
    sweep.to_csv("results/multichip_latency_sweep.csv", index=False, float_format="%.4g")
    for hop in [0.5e-6, 1e-6, 5e-6]:
        r = [tp_step(m, kv_formats["int4"], h, 32, 32768, 8, i, 2, hop)["time"] * 1e3 for i in [False, True]]
        print("hop %.1f us: int4 ring %.2f ms, in-network %.2f ms" % (hop * 1e6, r[0], r[1]))
