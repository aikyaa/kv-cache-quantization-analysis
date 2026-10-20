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
    "fp16": dict(bits=16, group=0, meta=0),
    "int8": dict(bits=8, group=128, meta=2),
    "int4": dict(bits=4, group=64, meta=4),
}

# public peak specs (dense fp16)
hw = {
    "A100": dict(flops=312e12, bw=2.039e12, mem=80e9),
    "H100": dict(flops=989e12, bw=3.35e12, mem=80e9),
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
    flops = 2 * batch * P + 4 * batch * m["layers"] * m["heads"] * m["head_dim"] * ctx
    return weight_bytes, kv_bytes, flops


if __name__ == "__main__":
    for name, m in models.items():
        print(name, "params = %.2fB" % (num_params(m) / 1e9))
        for fname, f in kv_formats.items():
            kvt = kv_bytes_per_token(m, f)
            print("   %s: %.0f KB per token" % (fname, kvt / 1024))

    print()
    m = models["llama2-7b"]
    for ctx in [512, 2048, 8192, 32768]:
        row = []
        for fname, f in kv_formats.items():
            w, kv, fl = decode_step(m, f, 16, ctx)
            row.append("%s %.1f GB" % (fname, (w + kv) / 1e9))
        print("llama2-7b b=16 ctx=%d:" % ctx, " | ".join(row))
