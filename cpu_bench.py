import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from quant import pack_int4

# real decode attention on this laptop's cpu: does a smaller cache actually run faster?
# one llama-3-8b layer: 8 kv heads, 4 query heads each, head_dim 128

torch.set_num_threads(4)
KV_HEADS, GROUP, HD = 8, 4, 128
BLOCK = 4096


def bandwidth():
    x = torch.empty(256 * 1024 * 1024 // 4)  # 256 MB
    y = torch.empty_like(x)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        y.copy_(x)
        best = min(best, time.perf_counter() - t0)
    return 2 * x.numel() * 4 / best


def make_cache(L, fmt):
    k = torch.randn(KV_HEADS, L, HD)
    v = torch.randn(KV_HEADS, L, HD)
    if fmt == "fp32":
        return dict(k=k, v=v), k.numel() * 4 * 2
    if fmt == "fp16":
        return dict(k=k.half(), v=v.half()), k.numel() * 2 * 2
    if fmt == "int8":
        ks = k.abs().amax(-1, keepdim=True) / 127
        vs = v.abs().amax(-1, keepdim=True) / 127
        c = dict(k=torch.round(k / ks).to(torch.int8), v=torch.round(v / vs).to(torch.int8),
                 ks=ks.half(), vs=vs.half())
        return c, k.numel() * 2 + 2 * ks.numel() * 2
    if fmt == "int4":
        # pack per block so a block can be unpacked on its own
        c = dict(k=[], v=[])
        nbytes = 0
        for i in range(0, L, BLOCK):
            for name, x in [("k", k), ("v", v)]:
                p, s, mn, shape = pack_int4(x[:, i:i + BLOCK].contiguous(), 64)
                c[name].append((p, s, mn, shape))
                nbytes += p.numel() + 2 * s.numel() + 2 * mn.numel()
        return c, nbytes


def get_block(c, fmt, name, i, j):
    if fmt == "fp32":
        return c[name][:, i:j]
    if fmt == "fp16":
        return c[name][:, i:j].float()
    if fmt == "int8":
        return c[name][:, i:j].float() * c[name + "s"][:, i:j].float()
    p, s, mn, shape = c[name][i // BLOCK]
    lo = (p & 15).float()
    hi = (p >> 4).float()
    q = torch.stack([lo, hi], 1).reshape(-1, 64)
    return (q * s.float() + mn.float()).reshape(shape)


def attention(q, c, fmt, L):
    # flash-decoding style: go through the cache in blocks with a running softmax
    m = torch.full((KV_HEADS, GROUP, 1), -1e30)
    s = torch.zeros(KV_HEADS, GROUP, 1)
    acc = torch.zeros(KV_HEADS, GROUP, HD)
    for i in range(0, L, BLOCK):
        j = min(i + BLOCK, L)
        k = get_block(c, fmt, "k", i, j)
        v = get_block(c, fmt, "v", i, j)
        sc = torch.bmm(q, k.transpose(1, 2)) / HD ** 0.5
        m_new = torch.maximum(m, sc.amax(-1, keepdim=True))
        p = torch.exp(sc - m_new)
        corr = torch.exp(m - m_new)
        s = s * corr + p.sum(-1, keepdim=True)
        acc = acc * corr + torch.bmm(p, v)
        m = m_new
    return acc / s


if __name__ == "__main__":
    bw = bandwidth()
    print("copy bandwidth %.1f GB/s" % (bw / 1e9))
    rows = []
    q = torch.randn(KV_HEADS, GROUP, HD)
    for L in [4096, 16384, 65536, 131072]:
        ref = None
        for fmt in ["fp32", "fp16", "int8", "int4"]:
            c, nbytes = make_cache(L, fmt)
            attention(q, c, fmt, L)
            times = []
            for _ in range(7):
                t0 = time.perf_counter()
                attention(q, c, fmt, L)
                times.append(time.perf_counter() - t0)
            best = sorted(times)[3]
            if fmt == "fp32":
                ref = best
            rows.append(dict(ctx=L, kv=fmt, MB=nbytes / 1e6, ms=best * 1e3,
                             GBps=nbytes / best / 1e9, speedup_vs_fp32=ref / best,
                             bytes_ratio_vs_fp32=nbytes / (L * KV_HEADS * HD * 8)))
            print(rows[-1], flush=True)
            del c
    df = pd.DataFrame(rows)
    df.to_csv("results/cpu_bench.csv", index=False, float_format="%.4g")

    colors = {"fp32": "#6250d6", "fp16": "#2a78d6", "int8": "#eb6834", "int4": "#1baf7a"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e5e5"})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    big = df[df.ctx == df.ctx.max()]
    ax.scatter(1 / big.bytes_ratio_vs_fp32, big.speedup_vs_fp32, s=50,
               color=[colors[k] for k in big.kv], zorder=3)
    for _, r in big.iterrows():
        ax.annotate(r.kv, (1 / r.bytes_ratio_vs_fp32, r.speedup_vs_fp32), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)
    ax.plot([1, 8], [1, 8], color="gray", ls=":", lw=1)
    ax.text(5.2, 5.6, "speedup = byte savings", color="gray", fontsize=8, rotation=30)
    ax.set_xlabel("cache is this many times smaller than fp32")
    ax.set_ylabel("attention speedup over fp32")
    ax.set_title("one Llama-3-8B layer on this laptop's CPU, ctx 128k")
    plt.tight_layout()
    plt.savefig("results/figures/cpu_bench.png", dpi=150)
