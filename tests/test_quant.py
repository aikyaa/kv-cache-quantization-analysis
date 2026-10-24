import torch

from quant import quantize, quantize_kv, pack_int4, unpack_int4


def test_int8_error_is_small():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    err = (quantize(x, 8, 64) - x).abs().max()
    assert err < (x.max() - x.min()) / 255


def test_pack_unpack_matches_fake_quant():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 10, 64)
    packed, scale, mn, shape = pack_int4(x, 64)
    assert packed.dtype == torch.uint8
    assert packed.numel() == x.numel() // 2
    back = unpack_int4(packed, scale, mn, shape, 64)
    assert torch.allclose(back, quantize_kv(x, 4, 64), atol=1e-2)


def test_per_token_quant_is_idempotent():
    # matters for decode: the cache gets re-quantized every step
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 64)
    once = quantize_kv(x, 4, 64)
    twice = quantize_kv(once, 4, 64)
    assert torch.allclose(once, twice, atol=1e-5)


def test_channel_axis_keeps_shape_with_odd_length():
    x = torch.randn(1, 2, 37, 64)
    assert quantize_kv(x, 4, 32, axis="channel").shape == x.shape


def test_16_bits_is_just_fp16():
    x = torch.randn(1, 1, 4, 8)
    assert torch.equal(quantize_kv(x, 16, 64), x.half().float())
