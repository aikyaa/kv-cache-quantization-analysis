import pandas as pd
import matplotlib.pyplot as plt

from ppl import configs

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e5e5"})


def bpv(bits, group):
    if bits >= 16:
        return 2.0
    return bits / 8 + 4 / group  # fp16 scale + min per group


df = pd.read_csv("results/ppl_gpt2.csv")
df = df[df.kv != "fp32 (no quant)"]
df["bytes_per_value"] = [(bpv(c["k_bits"], c["k_group"]) + bpv(c["v_bits"], c["v_group"])) / 2
                         for c in (configs[k] for k in df.kv)]
df["traffic_vs_fp16"] = df.bytes_per_value / 2.0
df.to_csv("results/ppl_vs_bytes.csv", index=False, float_format="%.4f")
print(df.to_string())

base = df[df.kv == "fp16"].ppl.iloc[0]
fig, ax = plt.subplots(figsize=(7, 4.5))
plot = df[df.ppl < 60]
ax.scatter(plot.traffic_vs_fp16, plot.ppl, s=40, color="#2a78d6", zorder=3)
for _, r in plot.iterrows():
    off = (6, -12) if r.kv == "int4 K per-channel" else (6, 3)
    ax.annotate(r.kv, (r.traffic_vs_fp16, r.ppl), textcoords="offset points", xytext=off, fontsize=7)
ax.axhline(base, color="gray", ls=":", lw=1)
ax.set_xlabel("KV bytes moved vs fp16")
ax.set_ylabel("WikiText-2 perplexity")
ax.set_title("GPT-2 small, quantized KV (int2 g32 = %.0f, off chart)" % df[df.kv == "int2 g32"].ppl.iloc[0])
plt.tight_layout()
plt.savefig("results/figures/ppl_vs_bytes.png", dpi=150)
