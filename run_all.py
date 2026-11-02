import sys
import time
import subprocess

# reruns everything in order. the slow stuff only runs with --all
# python run_all.py        ~5 min
# python run_all.py --all  a few hours on a laptop

fast = [
    ["roofline.py"],
    ["plots.py"],
    ["multichip.py"],
    ["quant.py"],
    ["cpu_bench.py"],
    ["fused_bench.py"],
    ["-m", "pytest", "-q"],
]

slow = [
    ["crosscheck.py"],
    ["compare_sim.py"],
    ["dataflow.py"],
    ["ppl.py", "gpt2", "1024", "65536"],
    ["ppl.py", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "2048", "16384"],
    ["plot_ppl.py"],
    ["decode_check.py"],
    ["kivi.py"],
]

steps = fast + slow if "--all" in sys.argv else fast
for s in steps:
    print("==>", " ".join(s), flush=True)
    t0 = time.time()
    subprocess.run([sys.executable] + s, check=True)
    print("    took %.0fs" % (time.time() - t0), flush=True)
