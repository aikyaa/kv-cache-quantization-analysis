import os
import sys
import pandas as pd
from scalesim.scale_sim import scalesim

cfg_text = """[general]
run_name = {name}

[architecture_presets]
ArrayHeight : {rows}
ArrayWidth : {cols}
IfmapSramSzkB : {sram}
FilterSramSzkB : {sram}
OfmapSramSzkB : {sram}
IfmapOffset : 0
FilterOffset : 100000000
OfmapOffset : 200000000
Bandwidth : {bw}
Dataflow : {df}
ReadRequestBuffer : 32
WriteRequestBuffer : 32

[layout]
IfmapCustomLayout : False
FilterCustomLayout : False
IfmapSRAMBankBandwidth : {bank_bw}
IfmapSRAMBankNum : 8
IfmapSRAMBankPort : 2
FilterSRAMBankBandwidth : {bank_bw}
FilterSRAMBankNum : 8
FilterSRAMBankPort : 2

[sparsity]
SparsitySupport : false

[run_presets]
InterfaceBandwidth : USER
UseRamulatorTrace : False
"""


def run_gemm(name, M, N, K, rows=128, cols=128, bw=256, sram=8192, df="ws", folder="sim_tmp"):
    os.makedirs(folder, exist_ok=True)
    cfg = folder + "/" + name + ".cfg"
    topo = folder + "/" + name + ".csv"
    lay = folder + "/" + name + "_layout.csv"
    with open(cfg, "w") as f:
        f.write(cfg_text.format(name=name, rows=rows, cols=cols, sram=sram, bw=bw,
                                bank_bw=bw // 8, df=df))
    with open(topo, "w") as f:
        f.write("Layer, M, N, K,\n%s, %d, %d, %d,\n" % (name, M, N, K))
    with open(lay, "w") as f:
        f.write("Layer, a, b, c, d, e, f, g, h, i,\n%s, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n" % name)

    s = scalesim(save_disk_space=True, verbose=False, config=cfg, topology=topo,
                 layout=lay, input_type_gemm=True)
    s.run_scale(top_path=folder + "/out")

    out = folder + "/out/" + name + "/"
    comp = pd.read_csv(out + "COMPUTE_REPORT.csv", skipinitialspace=True)
    acc = pd.read_csv(out + "DETAILED_ACCESS_REPORT.csv", skipinitialspace=True)
    return dict(name=name, M=M, N=N, K=K, bw=bw, sram=sram,
                cycles=int(comp["Total Cycles"][0]),
                stall=int(comp["Stall Cycles"][0]),
                util=float(comp["Compute Util %"][0]),
                dram_ifmap=int(acc["DRAM IFMAP Reads"][0]),
                dram_filter=int(acc["DRAM Filter Reads"][0]),
                dram_ofmap=int(acc["DRAM OFMAP Writes"][0]))


if __name__ == "__main__":
    name, M, N, K, bw, sram = sys.argv[1], *map(int, sys.argv[2:7])
    r = run_gemm(name, M, N, K, bw=bw, sram=sram)
    print(r)
    pd.DataFrame([r]).to_csv("sim_tmp/" + name + "_result.csv", index=False)
