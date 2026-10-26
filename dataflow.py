import os
import pandas as pd
from multiprocessing import Pool

from crosscheck import make_jobs, ROWS, COLS, SRAM_KB

# same gemms as crosscheck.py but output and input stationary

PICK = ["qkv_b1", "qkv_b16", "qkv_b64", "down_b16",
        "qk_fp16_L2048", "qk_int4_L2048", "pv_fp16_L2048", "pv_int4_L2048"]


def run_one(job):
    from simulate import run_gemm
    name, M, N, K, bw, df = job
    tag = name + "_" + df
    if os.path.exists("sim_tmp/" + tag + "_result.csv"):
        return pd.read_csv("sim_tmp/" + tag + "_result.csv").iloc[0].to_dict()
    r = run_gemm(tag, M, N, K, rows=ROWS, cols=COLS, bw=bw, sram=SRAM_KB, df=df)
    r["gemm"] = name
    r["dataflow"] = df
    pd.DataFrame([r]).to_csv("sim_tmp/" + tag + "_result.csv", index=False)
    print("finished", tag, flush=True)
    return r


if __name__ == "__main__":
    jobs = [j + (df,) for j in make_jobs() if j[0] in PICK for df in ["os", "is"]]
    with Pool(8) as p:
        res = pd.DataFrame(p.map(run_one, jobs, chunksize=1))

    ws = pd.read_csv("results/scalesim_compare.csv").set_index("gemm")
    rows = []
    for g in PICK:
        r = dict(gemm=g, roofline=ws.loc[g, "roofline_cycles"], ws=ws.loc[g, "sim_cycles"])
        for df in ["os", "is"]:
            r[df] = res[(res.gemm == g) & (res.dataflow == df)].cycles.iloc[0]
        r["best"] = min(r["ws"], r["os"], r["is"])
        r["best_over_roofline"] = r["best"] / r["roofline"]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv("results/scalesim_dataflows.csv", index=False, float_format="%.4g")
    print(out.to_string())
