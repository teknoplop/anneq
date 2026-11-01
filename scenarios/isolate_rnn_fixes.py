"""Isolate the RNN fixes #10 (RNN-FFWD) and #11 (RNN-KRON) on the telephone channel, each with its
learning rate swept. Writes JSON to argv[1]; results/RNN_ISOLATION.md summarises it.
Usage: uv run --no-project python scenarios/isolate_rnn_fixes.py out.json
"""
import sys, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_fixed as rf, run_scenarios as rs
CFG = {"10only": {"EQ_FIX": "RNN-FFWD,RETRY-CAP"}, "10+11": {"EQ_FIX": "RNN-FFWD,RNN-KRON,RETRY-CAP"}, "none": {"EQ_FIX": "RETRY-CAP"}}
ETAS = {"10only": (0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02),
        "10+11": (0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2),
        "none": (0.001, 0.002, 0.003, 0.005, 0.008, 0.012)}
def job(a):
    c, eta, snr = a
    p = dict(ch=rs.REAL, inputs=10, nodes=4, delay=5, eta=eta, snr=snr)
    r = rf.run_one("RNN", p, 40, CFG[c])
    return c, eta, snr, r["ber"]
if __name__ == "__main__":
    work = [(c, e, s) for c in CFG for e in ETAS[c] for s in (16, 18)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(job, work))
    json.dump(res, open(sys.argv[1], "w"))
    for r in res: print(*r)
