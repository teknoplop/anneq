"""Rerun the report's scenarios with the fixed Python line port (issues #10-#18).

uv run --no-project python scenarios/run_fixed.py [--only SUBSTR] [--jobs 8] [--sweep]
Same jobs as run_scenarios.py (imported from there). Output in the same format, to
results/fixed/<scenario>.csv and results/fixed/curves/, so scenarios/plot.py --out fixed reads it.
Implementations:
  pyfix       src/anneq_fixed.py with every fix (its default; see the tags there)
  pyfix_pn31  the same but keeping the 1995 31-bit PN preamble (EQ_NOFIX=RAND-PREAMBLE), to
              separate the effect of the training data from the code fixes
Extra CSV columns: final_err_all = final error averaged over every trial's last attempt, passed or
not (curves/<scen>__<impl>_all__<setting>.txt when some trials gave up); gaveup_per_trial: share of trials that failed all EQ_MAX_ATTEMPTS (10) training
attempts (RETRY-CAP). fail_per_trial is failed attempts per trial, as before.
RNN learning rates are re-tuned for the RNN-KRON fix (#11) from --sweep (results/fixed/rnn_eta_sweep.csv):
see RNN_ETA_FIXED. The numpy port is not used: it has its own random generator, so it is not
bit-exact with the C/line ports.
REMOTE=host: rsync src/anneq_*.py + scenarios/*.py to ~/anneq-runs/pyfix/ on host, run there, rsync results back.
"""
import argparse, concurrent.futures as cf, csv, os, pathlib, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scenarios"))
import run_scenarios as rs  # noqa: E402

OUT = ROOT / "results" / "fixed"
SEED = rs.SEED
# 1995: toy channel 0.005 with 3 nodes, telephone 0.003 with 4 nodes. RNN-KRON divides the
# delta*u term by nodes_1, so the old effective rate is eta*nodes_1 (0.015 / 0.012). Picked from the
# sweep (lowest mean BER over the sweep SNRs, 20 trials): see RESULTS_FIXED.md.
RNN_ETA_FIXED = {"toy": 0.12, "tel": 0.08}
IMPLS = {"pyfix": {}, "pyfix_pn31": {"EQ_NOFIX": "RAND-PREAMBLE"}}   # anneq_fixed.py: all fixes by default
ADA_SWEEP_ETAS = (0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024, 0.032)
SWEEP_ETAS = (0.003, 0.006, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5)


def fixed_params(prog, p):
    p = dict(p)
    if prog == "RNN":
        p["eta"] = RNN_ETA_FIXED["tel" if len(p["ch"]) > 2 else "toy"]
    return p


def write_job(d, prog, p, runs):
    d.mkdir(parents=True, exist_ok=True)
    w = "".join(f"{x}\n" for x in p["ch"])
    name = "BP_CONST.TXT" if prog == "BPBC" else f"{prog}_CON.TXT"   # BP-PATHS fix: no e:\temp\
    (d / name).write_text(rs.con_line(prog, p, runs))
    (d / "WEIGHTS.TXT").write_text(w)


def run_one(prog, p, runs, env_extra):
    py = ROOT / "src" / "anneq_fixed.py"
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        write_job(d, prog, p, runs)
        env = dict(os.environ, EQ_SEED=str(SEED), **env_extra)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(py), prog, "TEST"], cwd=d, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, timeout=6 * 3600)
        dt = time.time() - t0

        def f(n):
            q = d / n
            return [float(v) for v in q.read_text().split()] if q.exists() else []
        ber, fail, gave, curve = f("BER.TXT"), f("FAIL.TXT"), f("GAVEUP.TXT"), f("ERRORS.TXT")
        curve_all = f("ERRORS_ALL.TXT")
        if r.returncode != 0:
            print(f"job failed ({prog} {p}): {r.stderr[-500:]}", flush=True)
        return dict(ber=ber[0] if ber else None, fail=fail[0] if fail else None,
                    gaveup=gave[0] if gave else None, curve=curve, curve_all=curve_all, hung=not ber, secs=dt)


def write_csv(results, out):
    (out / "curves").mkdir(parents=True, exist_ok=True)
    rows = {}
    for (scen, impl, lab), (r, n) in sorted(results.items()):
        c = r["curve"]
        rows.setdefault(scen, []).append(dict(
            impl=impl, setting=lab, trials=n, ber=r["ber"], fail_per_trial=r["fail"],
            final_err=(sum(c[-100:]) / 100 if len(c) >= 100 else None), hung=r["hung"],
            gaveup_per_trial=r["gaveup"], secs=round(r["secs"], 1),
            final_err_all=(sum(r["curve_all"][-100:]) / 100 if len(r["curve_all"]) >= 100 else None)))
        if c:
            (out / "curves" / f"{scen}__{impl}__{lab}.txt").write_text("\n".join(f"{v:.7g}" for v in c) + "\n")
        if r["curve_all"] and r["gaveup"]:   # curve over all trials incl. the ones that never passed
            (out / "curves" / f"{scen}__{impl}_all__{lab}.txt").write_text(
                "\n".join(f"{v:.7g}" for v in r["curve_all"]) + "\n")
    for scen, rr in rows.items():   # merge: new rows replace same (impl, setting)
        old = out / f"{scen}.csv"
        if old.exists():
            new = {(r["impl"], r["setting"]) for r in rr}
            rr = [r for r in csv.DictReader(open(old)) if (r["impl"], r["setting"]) not in new] + rr
        with open(old, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[-1]))
            w.writeheader(); w.writerows(rr)


def remote(host, argv):
    base = "anneq-runs/pyfix"
    subprocess.run(["ssh", host, f"mkdir -p ~/{base}/src ~/{base}/scenarios ~/{base}/results/fixed"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/src/anneq_orig.py", f"{ROOT}/src/anneq_fixed.py", f"{ROOT}/src/anneq_numpy.py", f"{host}:{base}/src/"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/scenarios/run_fixed.py", f"{ROOT}/scenarios/run_scenarios.py",
                    f"{host}:{base}/scenarios/"], check=True)
    if OUT.exists():
        subprocess.run(["rsync", "-a", f"{OUT}/", f"{host}:{base}/results/fixed/"], check=True)
    args = " ".join(a for a in argv)
    subprocess.run(["ssh", host, f"cd ~/{base} && /opt/homebrew/bin/uv run --no-project python "
                                 f"scenarios/run_fixed.py {args}"], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{host}:{base}/results/fixed/", f"{OUT}/"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--impls", default=",".join(IMPLS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--sweep", action="store_true", help="RNN learning-rate sweep instead of the scenarios")
    a = ap.parse_args()
    if os.environ.get("REMOTE"):
        host = os.environ.pop("REMOTE")
        return remote(host, sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    work = []
    if a.sweep:
        for fam, ch, base, snrs in (("toy", rs.CH1, dict(inputs=2, nodes=3, delay=0), (10, 14)),
                                    ("tel", rs.REAL, dict(inputs=10, nodes=4, delay=5), (14, 18))):
            for eta in SWEEP_ETAS:
                for s in snrs:
                    p = dict(base, ch=ch, eta=eta, snr=s)
                    work.append(((f"rnn_eta_sweep_{fam}", "pyfix", f"eta={eta},snr={s}"), "RNN", p, 20,
                                 IMPLS["pyfix"]))
        # fairness check: the telephone ADALINE (15 taps, delay 5) over its own stable range
        for eta in ADA_SWEEP_ETAS:
            for s in (14, 18):
                p = dict(ch=rs.REAL, taps=15, delay=5, eta=eta, snr=s)
                work.append((("ada_eta_sweep_tel", "pyfix", f"eta={eta},snr={s}"), "LMS", p, 20, IMPLS["pyfix"]))
    else:
        for scen, prog, lab, p in rs.JOBS:
            if a.only not in scen:
                continue
            n = 20 if prog == "BPBC" else rs.TRIALS
            for impl in a.impls.split(","):
                work.append(((scen, impl, lab), prog, fixed_params(prog, p), n, IMPLS[impl]))
    # longest first (telephone RNN, BPBC) so the pool stays busy
    work.sort(key=lambda w: (w[1] != "RNN" or len(w[2]["ch"]) < 3, w[1] != "BPBC"))
    if a.sweep:   # only ADALINE rows are new if the RNN sweep is already there
        done = {r["setting"] for r in csv.DictReader(open(OUT / "rnn_eta_sweep_tel.csv"))} \
            if (OUT / "rnn_eta_sweep_tel.csv").exists() and os.environ.get("EQ_SWEEP_ALL") != "1" else set()
        work = [w for w in work if not (w[0][0].startswith("rnn_") and done)]
    t0 = time.time()
    results = {}
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run_one, prog, p, n, env): (key, n) for key, prog, p, n, env in work}
        for k, f in enumerate(cf.as_completed(futs), 1):
            key, n = futs[f]
            results[key] = (f.result(), n)
            print(f"[{k}/{len(work)}] {key} {results[key][0]['secs']:.0f} s ber={results[key][0]['ber']}", flush=True)
    print(f"Python fixed: {len(work)} jobs in {time.time()-t0:.0f} s, {a.jobs} workers", flush=True)
    write_csv(results, OUT)


if __name__ == "__main__":
    main()
