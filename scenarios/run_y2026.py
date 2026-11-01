"""2026 re-run (unprefixed modules in src/): run the sweeps in parallel and write results/y2026/<scenario>.csv.

    uv run scenarios/run_y2026.py [offline|preamble|all] [--jobs N] [--quick]

Set Y2026_HOST=<ssh host> to run on another machine instead (needs uv there); the package
(pyproject.toml, uv.lock, src, scenarios) is rsynced to ~/anneq-runs/y2026/ on that host and results are copied back.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "y2026"


def _offline(args):
    from experiments import offline
    ch, snr, c = args
    return offline(ch, snr, n_train=c["n_train"], n_val=c["n_val"], min_errors=c["min_errors"],
                   max_bits=c["max_bits"])


def _preamble(args):
    from experiments import preamble
    ch, snr, n, t, c = args
    return preamble(ch, snr, n, t, n_test=c["n_test"], steps=c["steps"])


def tasks(scn, cfg, quick):
    c = cfg[scn]
    if scn == "offline":
        if quick:
            c = dict(c, n_train=20000, max_bits=200000)
        return _offline, [(ch, s, c) for ch, snrs in c["snr_db"].items()
                          for s in (snrs[::4] if quick else snrs)]
    if quick:
        c = dict(c, trials=1, n_pre=c["n_pre"][::3], steps=300)
    return _preamble, [(ch, s, n, t, c) for ch, s in c["snr_db"].items() for n in c["n_pre"]
                       for t in range(c["trials"])]


def write(rows, path):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def remote(host, argv):
    dst = "~/anneq-runs/y2026/"
    subprocess.run(["ssh", host, "mkdir -p " + dst], check=True)
    subprocess.run(["ssh", host, f"mkdir -p {dst}src {dst}scenarios"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/pyproject.toml", f"{ROOT}/uv.lock", f"{host}:{dst}"], check=True)
    for d in ("src", "scenarios"):
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", f"{ROOT}/{d}/", f"{host}:{dst}{d}/"],
                       check=True)
    subprocess.run(["ssh", host, f"cd {dst} && PATH=/opt/homebrew/bin:~/.local/bin:$PATH uv run scenarios/run_y2026.py " +
                    " ".join(argv)], check=True)
    subprocess.run(["rsync", "-a", f"{host}:{dst}results/y2026/", f"{RES}/"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["offline", "preamble", "all"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "scenarios/configs/y2026.yaml"))
    a = ap.parse_args()
    host = os.environ.get("Y2026_HOST")
    if host:
        os.environ.pop("Y2026_HOST")
        return remote(host, sys.argv[1:])
    cfg = yaml.safe_load(open(a.config))
    RES.mkdir(parents=True, exist_ok=True)
    for scn in (["offline", "preamble"] if a.scenario == "all" else [a.scenario]):
        fn, ts = tasks(scn, cfg, a.quick)
        t0, rows = time.time(), []
        with Pool(a.jobs) as p:
            for i, r in enumerate(p.imap_unordered(fn, ts), 1):
                rows += r
                print(f"{scn}: {i}/{len(ts)} done, {time.time() - t0:.0f}s", flush=True)
        write(rows, RES / f"{scn}{'_quick' if a.quick else ''}.csv")


if __name__ == "__main__":
    main()
