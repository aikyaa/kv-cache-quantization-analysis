import os
import pandas as pd
from multiprocessing import Pool

from roofline import models, kv_formats, bytes_per_value

# simulated chip: 128x128 ws array at 1 GHz, 512 B/cycle off chip (= 256 fp16 words)
ROWS = 128
COLS = 128
BYTES_PER_CYCLE = 512
SRAM_KB = 32768

m = models["llama3.2-1b"]
group = m["heads"] // m["kv_heads"]
d, hd = m["d"], m["head_dim"]


def make_jobs():
    jobs = []
    for b in [1, 16, 64]:
        jobs.append(("qkv_b%d" % b, b, (m["heads"] + 2 * m["kv_heads"]) * hd, d, 256))
        jobs.append(("o_b%d" % b, b, d, m["heads"] * hd, 256))
        jobs.append(("upgate_b%d" % b, b, 2 * m["ff"], d, 256))
        jobs.append(("down_b%d" % b, b, d, m["ff"], 256))
    for ctx in [512, 2048, 8192]:
        for fname, f in kv_formats.items():
            # scalesim has no data types, so give it more words/cycle for narrow kv
            words = int(BYTES_PER_CYCLE / bytes_per_value(f)) // 64 * 64
            jobs.append(("qk_%s_L%d" % (fname, ctx), group, ctx, hd, words))
            jobs.append(("pv_%s_L%d" % (fname, ctx), group, hd, ctx, words))
    return jobs


def run_one(job):
    from simulate import run_gemm
    name, M, N, K, bw = job
    if os.path.exists("sim_tmp/" + name + "_result.csv"):
        return pd.read_csv("sim_tmp/" + name + "_result.csv").iloc[0].to_dict()
    r = run_gemm(name, M, N, K, rows=ROWS, cols=COLS, bw=bw, sram=SRAM_KB)
    pd.DataFrame([r]).to_csv("sim_tmp/" + name + "_result.csv", index=False)
    print("finished", name, flush=True)
    return r


if __name__ == "__main__":
    jobs = make_jobs()
    with Pool(10) as p:
        results = p.map(run_one, jobs, chunksize=1)
    pd.DataFrame(results).to_csv("results/scalesim_runs.csv", index=False)
    print("done")
