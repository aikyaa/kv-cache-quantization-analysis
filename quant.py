import torch


def quantize(x, bits, group, sym=False):
    # fake quant along the last dim in groups, returns dequantized tensor
    shape = x.shape
    x = x.reshape(-1, group).float()
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = x.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)
        q = torch.round(x / scale).clamp(-qmax - 1, qmax)
        out = q * scale
    else:
        qmax = 2 ** bits - 1
        mn = x.amin(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp(min=1e-8)
        q = torch.round((x - mn) / scale).clamp(0, qmax)
        out = q * scale + mn
    return out.reshape(shape)


def quantize_kv(x, bits, group, axis="token", sym=False):
    # x is (batch, heads, tokens, head_dim)
    # axis="token": groups run along head_dim, one token at a time
    # axis="channel": groups run along tokens, one channel at a time (KIVI keys)
    if bits >= 16:
        return x.half().to(x.dtype)
    if axis == "token":
        return quantize(x, bits, group, sym).to(x.dtype)
    T = x.shape[2]
    pad = (-T) % group
    xt = x.transpose(2, 3)
    if pad:
        xt = torch.cat([xt, xt[..., -1:].expand(*xt.shape[:-1], pad)], dim=-1)
    out = quantize(xt, bits, group, sym)[..., :T]
    return out.transpose(2, 3).to(x.dtype)


def pack_int4(x, group):
    # real int4 storage: two values per byte + fp16 scale and min per group
    shape = x.shape
    x = x.reshape(-1, group).float()
    mn = x.amin(dim=1, keepdim=True)
    mx = x.amax(dim=1, keepdim=True)
    scale = ((mx - mn) / 15).clamp(min=1e-8)
    q = torch.round((x - mn) / scale).clamp(0, 15).to(torch.uint8)
    q = q.reshape(-1, 2)
    packed = q[:, 0] | (q[:, 1] << 4)
    return packed, scale.half(), mn.half(), shape


def unpack_int4(packed, scale, mn, shape, group):
    lo = packed & 15
    hi = packed >> 4
    q = torch.stack([lo, hi], dim=1).reshape(-1, group).float()
    return (q * scale.float() + mn.float()).reshape(shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    # one gpt2 sized kv cache: 12 layers, 12 heads, 1024 tokens, 64 dims, k and v
    k = torch.randn(2 * 12, 12, 1024, 64)
    packed, scale, mn, shape = pack_int4(k, 64)
    nbytes = packed.numel() + 2 * scale.numel() + 2 * mn.numel()
    fp16_bytes = k.numel() * 2
    print("fp16 bytes", fp16_bytes)
    print("int4 bytes", nbytes, "-> %.4f bytes per value" % (nbytes / k.numel()))
    back = unpack_int4(packed, scale, mn, shape, 64)
    print("pack/unpack matches fake quant:", torch.allclose(back, quantize_kv(k, 4, 64), atol=1e-2))
