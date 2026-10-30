import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

# same attention as cpu_bench.py but each row is dequantized into a tiny buffer that stays in L1
# one llama-3-8b layer: 8 kv heads, 4 query heads each, head_dim 128

KV_HEADS, GROUP, HD = 8, 4, 128
CHUNKS = 16  # split each head's cache so all cores get work


@njit(fastmath=True, inline="always")
def row_fp32(c, s, z, h, t, buf):
    for d in range(HD):
        buf[d] = c[h, t, d]


@njit(fastmath=True, inline="always")
def row_int8(c, s, z, h, t, buf):
    sc = s[h, t, 0]
    for d in range(HD):
        buf[d] = c[h, t, d] * sc


@njit(fastmath=True, inline="always")
def row_int4(c, s, z, h, t, buf):
    for g in range(HD // 64):
        sc = s[h, t, g]
        mn = z[h, t, g]
        for i in range(32):
            byte = c[h, t, g * 32 + i]
            buf[g * 64 + 2 * i] = (byte & 15) * sc + mn
            buf[g * 64 + 2 * i + 1] = (byte >> 4) * sc + mn


def make_kernel(row):
    @njit(parallel=True, fastmath=True)
    def attn(q, kc, ks, kz, vc, vs, vz, L):
        part_m = np.zeros((KV_HEADS, CHUNKS, GROUP), dtype=np.float32)
        part_s = np.zeros((KV_HEADS, CHUNKS, GROUP), dtype=np.float32)
        part_acc = np.zeros((KV_HEADS, CHUNKS, GROUP, HD), dtype=np.float32)
        step = (L + CHUNKS - 1) // CHUNKS
        scale = np.float32(1.0 / np.sqrt(HD))
        for job in prange(KV_HEADS * CHUNKS):
            h = job // CHUNKS
            c = job % CHUNKS
            t0 = c * step
            t1 = min(t0 + step, L)
            buf = np.empty(HD, dtype=np.float32)  # one dequantized row, stays in L1
            sc = np.empty((GROUP, t1 - t0), dtype=np.float32)
            # pass 1: scores
            for t in range(t0, t1):
                row(kc, ks, kz, h, t, buf)
                for g in range(GROUP):
                    acc = np.float32(0.0)
                    for d in range(HD):
                        acc += q[h, g, d] * buf[d]
                    sc[g, t - t0] = acc * scale
            for g in range(GROUP):
                m = sc[g].max()
                tot = np.float32(0.0)
                for i in range(t1 - t0):
                    e = np.exp(sc[g, i] - m)
                    sc[g, i] = e
                    tot += e
                part_m[h, c, g] = m
                part_s[h, c, g] = tot
            # pass 2: weighted sum of values
            for t in range(t0, t1):
                row(vc, vs, vz, h, t, buf)
                for g in range(GROUP):
                    p = sc[g, t - t0]
                    for d in range(HD):
                        part_acc[h, c, g, d] += p * buf[d]
        return part_m, part_s, part_acc
    return attn


kernels = {"fp32": make_kernel(row_fp32), "int8": make_kernel(row_int8), "int4": make_kernel(row_int4)}


def combine(part_m, part_s, part_acc):
    m = part_m.max(axis=1, keepdims=True)
    w = np.exp(part_m - m)
    s = (part_s * w).sum(axis=1)
    acc = (part_acc * w[..., None]).sum(axis=1)
    return acc / s[..., None]


def make_cache(x, fmt):
    one = np.zeros((1, 1, 1), dtype=np.float32)
    if fmt == "fp32":
        return x, one, one, x.nbytes
    if fmt == "int8":
        s = np.abs(x).max(axis=-1, keepdims=True) / 127
        q = np.round(x / s).astype(np.int8)
        return q, s.astype(np.float32), one, q.nbytes + s.size * 2  # count scales as fp16
    g = x.reshape(KV_HEADS, -1, HD // 64, 64)
    mn = g.min(axis=-1)
    s = (g.max(axis=-1) - mn) / 15
    q = np.round((g - mn[..., None]) / s[..., None]).astype(np.uint8).reshape(KV_HEADS, -1, HD)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    return packed, s.astype(np.float32), mn.astype(np.float32), packed.nbytes + 2 * s.size * 2


def run(fmt, q, k, v, L, reps=15):
    kc, ks, kz, kb = make_cache(k, fmt)
    vc, vs, vz, vb = make_cache(v, fmt)
    f = kernels[fmt]
    out = combine(*f(q, kc, ks, kz, vc, vs, vz, L))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(q, kc, ks, kz, vc, vs, vz, L)
        times.append(time.perf_counter() - t0)
    return min(times), kb + vb, out


def reference(q, k, v):
    sc = np.einsum("hgd,htd->hgt", q, k) / np.sqrt(HD)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hgt,htd->hgd", p, v)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q = rng.standard_normal((KV_HEADS, GROUP, HD)).astype(np.float32)
    rows = []
    for L in [4096, 16384, 65536, 131072]:
        k = rng.standard_normal((KV_HEADS, L, HD)).astype(np.float32)
        v = rng.standard_normal((KV_HEADS, L, HD)).astype(np.float32)
        ref = reference(q, k, v)
        base = None
        for fmt in ["fp32", "int8", "int4"]:
            t, nbytes, out = run(fmt, q, k, v, L)
            if base is None:
                base = t
            err = np.abs(out - ref).max()
            rows.append(dict(ctx=L, kv=fmt, MB=nbytes / 1e6, ms=t * 1e3, GBps=nbytes / t / 1e9,
                             speedup_vs_fp32=base / t, bytes_ratio_vs_fp32=nbytes / (k.nbytes * 2),
                             max_err=err))
            print("ctx %6d %s  %7.2f ms  %5.1f GB/s  x%.2f  err %.3f" % (L, fmt, t * 1e3, nbytes / t / 1e9, base / t, err), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/fused_bench.csv", index=False, float_format="%.4g")

    unfused = pd.read_csv("results/cpu_bench.csv")
    colors = {"fp32": "#6250d6", "int8": "#eb6834", "int4": "#1baf7a"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e5e5"})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for fmt in ["int8", "int4"]:
        a = df[df.kv == fmt]
        b = unfused[unfused.kv == fmt]
        ax.plot(a.ctx, a.speedup_vs_fp32, "o-", color=colors[fmt], lw=2, label=fmt + " fused (numba)")
        ax.plot(b.ctx, b.speedup_vs_fp32, "s--", color=colors[fmt], lw=1.2, label=fmt + " unfused (pytorch)")
    ax.axhline(1, color="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("speedup over fp32 cache")
    ax.set_title("one Llama-3-8B layer of decode attention, laptop CPU")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/fused_vs_unfused.png", dpi=150)
