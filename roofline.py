import numpy as np

# model shapes
models = {
    "llama2-7b": dict(layers=32, d=4096, heads=32, kv_heads=32, head_dim=128, ff=11008, vocab=32000, gated=True),
    "llama3-8b": dict(layers=32, d=4096, heads=32, kv_heads=8, head_dim=128, ff=14336, vocab=128256, gated=True),
    "llama3.2-1b": dict(layers=16, d=2048, heads=32, kv_heads=8, head_dim=64, ff=8192, vocab=128256, gated=True),
    "gpt2": dict(layers=12, d=768, heads=12, kv_heads=12, head_dim=64, ff=3072, vocab=50257, gated=False),
}

# kv formats: bits per value, group size for scales, bytes of scale+zero per group
kv_formats = {
    "fp16": dict(bits=16, group=0, meta=0, dequant=0),
    "int8": dict(bits=8, group=128, meta=2, dequant=2),
    "int4": dict(bits=4, group=64, meta=4, dequant=3),
}

# public peak specs (dense fp16). vec = non tensor core flops
# "NPU" is made up: big matmul array, slow memory, tiny vector unit
hw = {
    "A100": dict(flops=312e12, vec=78e12, bw=2.039e12, mem=80e9),
    "H100": dict(flops=989e12, vec=134e12, bw=3.35e12, mem=80e9),
    "NPU": dict(flops=200e12, vec=4e12, bw=0.8e12, mem=64e9),
}


def num_params(m):
    d = m["d"]
    attn = 2 * d * m["heads"] * m["head_dim"] + 2 * d * m["kv_heads"] * m["head_dim"]
    if m["gated"]:
        mlp = 3 * d * m["ff"]
    else:
        mlp = 2 * d * m["ff"]
    return m["layers"] * (attn + mlp) + m["vocab"] * d


def bytes_per_value(f):
    b = f["bits"] / 8
    if f["group"] > 0:
        b += f["meta"] / f["group"]
    return b


def kv_bytes_per_token(m, f):
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bytes_per_value(f)


def decode_step(m, f, batch, ctx):
    P = num_params(m)
    weight_bytes = 2 * P
    kv_bytes = batch * (ctx + 1) * kv_bytes_per_token(m, f)
    act_bytes = batch * m["layers"] * 4 * m["d"] * 2
    flops = 2 * batch * P + 4 * batch * m["layers"] * m["heads"] * m["head_dim"] * ctx
    # dequant every kv value we read + softmax (~5 ops per score)
    kv_values = batch * ctx * 2 * m["layers"] * m["kv_heads"] * m["head_dim"]
    vec_ops = kv_values * f["dequant"] + 5 * batch * m["layers"] * m["heads"] * ctx
    return dict(weights=weight_bytes, kv=kv_bytes, act=act_bytes,
                bytes=weight_bytes + kv_bytes + act_bytes, flops=flops, vec_ops=vec_ops)


def step_time(m, f, h, batch, ctx):
    s = decode_step(m, f, batch, ctx)
    t_compute = s["flops"] / h["flops"]
    t_vec = s["vec_ops"] / h["vec"]
    t_mem = s["bytes"] / h["bw"]
    t = max(t_compute, t_vec, t_mem)
    if t == t_mem:
        bound = "memory"
    elif t == t_compute:
        bound = "compute"
    else:
        bound = "vector"
    return t, bound, t_compute, t_vec, t_mem


def crossover_batch(m, f, h, ctx):
    # step is linear in batch: bytes = W + B*b1, flops = B*f1
    # compute bound when B*f1 / (W + B*b1) >= ridge
    ridge = h["flops"] / h["bw"]
    s1 = decode_step(m, f, 1, ctx)
    W = s1["weights"]
    b1 = s1["bytes"] - W
    f1 = s1["flops"]
    if f1 - ridge * b1 <= 0:
        return float("inf")
    return ridge * W / (f1 - ridge * b1)


def max_ctx_compute_bound(m, f, h):
    # past this context no batch size is compute bound
    ridge = h["flops"] / h["bw"]
    a = decode_step(m, f, 1, 0)
    b = decode_step(m, f, 1, 1)
    b0 = a["bytes"] - a["weights"]
    f0 = a["flops"]
    db = b["bytes"] - a["bytes"]
    df = b["flops"] - a["flops"]
    if ridge * db - df <= 0:
        return float("inf")
    return (f0 - ridge * b0) / (ridge * db - df)


def max_batch(m, f, h, ctx):
    free = 0.9 * h["mem"] - 2 * num_params(m)
    return int(free // ((ctx + 1) * kv_bytes_per_token(m, f)))


if __name__ == "__main__":
    for name, m in models.items():
        print(name, "params = %.2fB" % (num_params(m) / 1e9))
        for fname, f in kv_formats.items():
            print("   %s: %.0f KB per token" % (fname, kv_bytes_per_token(m, f) / 1024))

    for hname, h in hw.items():
        print()
        print(hname, "ridge = %.0f flop/byte" % (h["flops"] / h["bw"]))
        for name in ["llama2-7b", "llama3-8b", "llama3.2-1b"]:
            m = models[name]
            for fname, f in kv_formats.items():
                Lmax = max_ctx_compute_bound(m, f, h)
                b2k = crossover_batch(m, f, h, 2048)
                print("  %-12s %-5s crossover B @2k = %7.1f   (fits %5d)   compute bound only below ctx %6.0f"
                      % (name, fname, b2k, max_batch(m, f, h, 2048), Lmax))
