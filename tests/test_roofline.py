import math
import pytest

from roofline import *
from multichip import allreduce, tp_step


def test_kv_bytes_llama2_7b():
    # 2 (k,v) x 32 layers x 32 heads x 128 dims x 2 bytes
    m = models["llama2-7b"]
    assert kv_bytes_per_token(m, kv_formats["fp16"]) == 2 * 32 * 32 * 128 * 2


def test_bytes_per_value():
    assert bytes_per_value(kv_formats["fp16"]) == 2
    assert bytes_per_value(kv_formats["int8"]) == 1 + 2 / 128
    assert bytes_per_value(kv_formats["int4"]) == 0.5 + 4 / 64


def test_param_count_llama2_7b():
    # 6.74B total minus the 32000 x 4096 input embedding we don't read
    assert num_params(models["llama2-7b"]) == pytest.approx(6.74e9 - 32000 * 4096, rel=0.01)


def test_crossover_balances_compute_and_memory():
    m, f, h = models["llama3-8b"], kv_formats["int4"], hw["A100"]
    b = crossover_batch(m, f, h, 512)
    s = decode_step(m, f, b, 512)
    assert s["flops"] / h["flops"] == pytest.approx(s["bytes"] / h["bw"], rel=1e-6)


def test_no_compute_bound_past_max_ctx():
    m, f, h = models["llama3-8b"], kv_formats["fp16"], hw["A100"]
    L = max_ctx_compute_bound(m, f, h)
    assert math.isfinite(crossover_batch(m, f, h, int(L) - 10))
    assert crossover_batch(m, f, h, int(L) + 10) == float("inf")


def test_quant_helps_more_at_long_context():
    m, h = models["llama3-8b"], hw["H100"]
    def speedup(ctx):
        return step_time(m, kv_formats["fp16"], h, 16, ctx)[0] / step_time(m, kv_formats["int4"], h, 16, ctx)[0]
    assert 1 < speedup(1024) < speedup(32768) < 4


def test_ring_allreduce_bytes():
    assert allreduce(100, 1, 1e9)[0] == 0
    assert allreduce(100, 4, 1e9)[0] == pytest.approx(150)
    assert allreduce(100, 4, 1e9, innet=True)[0] == 100


def test_one_chip_matches_single_chip_model():
    m, f, h = models["llama3-8b"], kv_formats["fp16"], hw["H100"]
    s = tp_step(m, f, h, 8, 4096, 1)
    single = decode_step(m, f, 8, 4096)
    assert s["t_link"] == 0
    # tp_step skips the small activation bytes
    assert s["t_hbm"] == pytest.approx((single["bytes"] - single["act"]) / h["bw"])
