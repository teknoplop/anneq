"""Run the 1995 report's scenarios on every implementation and write CSVs.

uv run --no-project --with numpy --with matplotlib python scenarios/run_scenarios.py [options]
  --impls clang,clang_fix,pyline,numpy,numpy_fix   (default: all of these)
  --only SUBSTR      only scenarios whose name contains SUBSTR
  --jobs N           parallel workers (default: cpu count)
  REMOTE=host        env var: run the C jobs on that host over ssh in ~/anneq-runs/
                     (needs clang + rsync there; sources are synced and built there)
Needs src/clang built locally (make -C src/clang all fixed) for the prepared sources.
Every C and py-line run uses EQ_SEED (deterministic seeds, see src/openwatcom/eqcompat.h), so they
agree bit for bit; numpy uses its own generator. DOS is not run here (see PROGRESS.md).
Output: results/rebuild/<scenario>.csv, results/rebuild/curves/<scenario>__<impl>__<setting>.txt
"""
import argparse, concurrent.futures as cf, csv, os, pathlib, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rebuild"
sys.path.insert(0, str(ROOT / "src"))

CH1 = [1.0, 0.5]
CH2 = [0.5, 1.0]
REAL = [0.047, 0.321, 0.783, 0.427, 0.293, 0.100, 0.041, 0.038, 0.020, 0.011]
SEED = 1000
TRIALS = 100
TIMEOUT = 90          # seconds per C job; LMS.C/RNN.C "while(fail)" retrain has no cap

JOBS = []
for t in (1, 2, 3, 7, 11):
    JOBS.append(("fig5_3", "LMS", f"taps={t}", dict(ch=CH1, taps=t, eta=0.01, snr=21, delay=0)))
for e in (0.1, 0.01, 0.001, 0.0001):
    JOBS.append(("fig5_4", "LMS", f"eta={e}", dict(ch=CH1, taps=7, eta=e, snr=21, delay=0)))
for s in (6, 8, 10, 12, 14):
    JOBS.append(("fig5_5", "LMS", f"snr={s}", dict(ch=CH1, taps=7, eta=0.01, snr=s, delay=0)))
    JOBS.append(("fig5_7", "LMS", f"snr={s}", dict(ch=CH2, taps=7, eta=0.01, snr=s, delay=7)))
for s in (6, 10, 14, 16, 18, 20):
    JOBS.append(("fig9_3_ada", "LMS", f"snr={s}", dict(ch=REAL, taps=15, eta=0.08, snr=s, delay=5)))
    # eta=0.08 as printed diverges; 0.008 reproduces Fig 9.3 -> probably a typo in the report
    JOBS.append(("fig9_3_ada008", "LMS", f"snr={s}", dict(ch=REAL, taps=15, eta=0.008, snr=s, delay=5)))
    JOBS.append(("fig9_3_rnn", "RNN", f"snr={s}", dict(ch=REAL, inputs=10, nodes=4, eta=0.003, snr=s, delay=5)))
JOBS.append(("fig7_2", "RNN", "snr=21", dict(ch=CH1, inputs=2, nodes=3, eta=0.005, snr=21, delay=0)))
for s in (6, 8, 10, 12, 14, 16, 18, 20):
    JOBS.append(("fig7_3", "RNN", f"snr={s}", dict(ch=CH1, inputs=2, nodes=3, eta=0.005, snr=s, delay=0)))
    JOBS.append(("fig8_3_ch1", "RB", f"snr={s}", dict(ch=CH1, inputs=2, nodes=8, width=0.125, snr=s)))
    JOBS.append(("fig8_3_ch2", "RB", f"snr={s}", dict(ch=CH2, inputs=2, nodes=8, width=0.125, snr=s)))
    # extra (not in the report): ADALINE with the RNN/RBF's 2 inputs, like-for-like on channel 1
    JOBS.append(("extra_ada2", "LMS", f"snr={s}", dict(ch=CH1, taps=2, eta=0.01, snr=s, delay=0)))
# extra: ADALINE with the RNN's 10 inputs on the telephone channel (like-for-like with fig9_3_rnn)
for s in (6, 10, 14, 16, 18, 20):
    JOBS.append(("extra_ada10_real", "LMS", f"snr={s}", dict(ch=REAL, taps=10, eta=0.008, snr=s, delay=5)))
# backprop, batch (BPBC.C): 2-6-6-1, 40 training samples, 1000 epochs. Learning rate and momentum
# are not given in the report: 0.1 / 0.5 assumed. Needs the BP-DEBUG fix to run at all on a hosted OS.
for ch, nm in ((CH1, "ch1"), (CH2, "ch2")):
    for s in (6, 10, 14, 21):
        JOBS.append((f"bp_{nm}", "BPBC", f"snr={s}", dict(ch=ch, bits=40, epochs=1000, snr=s, n=(2, 6, 6), lr=0.1, mom=0.5)))

VARIANTS = {"LMS": ("LMS-FILL", "LMS-BIAS"), "RNN": ("RNN-FFWD", "RNN-KRON")}
EXE = {"LMS": "LMS.EXE", "RNN": "RNN.EXE", "RB": "RB.EXE", "BPBC": "BPBC.EXE"}


def con_line(prog, p, runs):
    m = len(p["ch"])
    if prog == "LMS":
        return f"{m} 1000 {p['snr']} {runs} {p['delay']} {p['taps']} {p['eta']}\n"
    if prog == "RNN":
        return f"{m} 1000 {p['snr']} {runs} {p['delay']} {p['inputs']} {p['nodes']} {p['eta']}\n"
    if prog == "RB":
        return f"{m} 1000 {p['snr']} {runs} {p['inputs']} {p['nodes']} {p['width']}\n"
    n0, n1, n2 = p["n"]
    return f"{m} {p['bits']} {p['epochs']} {p['snr']} {runs} {n0} {n1} {n2} {p['lr']} {p['mom']}\n"


def write_job(d, prog, p, runs):
    d.mkdir(parents=True, exist_ok=True)
    w = "".join(f"{x}\n" for x in p["ch"])
    if prog == "BPBC":   # hard-coded e:\temp\ paths: on a hosted OS these are plain file names
        (d / "e:\\temp\\bp_const.txt").write_text(con_line(prog, p, runs))
        (d / "e:\\temp\\weights.txt").write_text(w)
    else:
        (d / f"{prog}_CON.TXT").write_text(con_line(prog, p, runs))
        (d / "WEIGHTS.TXT").write_text(w)


def read_results(d, prog):
    def f(n):
        q = d / n
        return [float(v) for v in q.read_text().split()] if q.exists() else []
    if prog == "BPBC":
        ber, fail, curve = f("e:\\temp\\ber.txt"), [], f("e:\\temp\\errors.txt")
    else:
        ber, fail, curve = f("BER.TXT"), f("FAIL.TXT"), f("ERRORS.TXT")
    hung = not ber
    return dict(ber=ber[0] if ber else None, fail=fail[0] if fail else None, curve=curve, hung=hung)


# Each C job dir holds the inputs plus the binary (or a link to it); run.sh runs it with a timeout.
# MallocNanoZone=1: the unfixed LMS writes one weight past a heap block (fill_rand); the src/clang heap
# guard absorbs it, this just keeps allocator behaviour the same everywhere.
RUNNER = """#!/bin/sh
cd "$1" || exit 1
EQ_SEED={seed} MallocNanoZone=1 perl -e 'alarm {timeout}; exec @ARGV' ./$(cat EXE) TEST >/dev/null 2>&1
exit 0
"""


def run_c_jobs(cjobs, workers):
    """cjobs: list of (key, impl, prog, p, runs). Returns {key: results}."""
    remote = os.environ.get("REMOTE")
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td) / "jobs"
        root.mkdir()
        (root / "run.sh").write_text(RUNNER.format(seed=SEED, timeout=TIMEOUT))
        os.chmod(root / "run.sh", 0o755)
        names = []
        for k, (key, impl, prog, p, runs) in enumerate(cjobs):
            d = root / f"J{k:04d}"
            write_job(d, prog, p, runs)
            (d / "EXE").write_text(EXE[prog] + "\n")
            bdir = ("build_fix" if impl == "clang_fix" or prog == "BPBC" else
                    "build_fix_" + impl[len("clang_fix_"):] if impl.startswith("clang_fix_") else "build")
            (d / "BIN").write_text(f"{bdir}/{prog.lower()}\n")
            names.append(d.name)
        (root / "list.txt").write_text("\n".join(names) + "\n")
        link = ('for j in $(cat list.txt); do ln -sf "$BIN/$(cat $j/BIN)" "$j/$(cat $j/EXE)"; done; '
                f'cat list.txt | xargs -P {workers} -n 1 ./run.sh')
        t0 = time.time()
        if remote:
            pc = ROOT / "src/clang"
            base = "anneq-runs"
            subprocess.run(["ssh", remote, f"mkdir -p ~/{base}/src ~/{base}/jobs"], check=True)
            subprocess.run(["rsync", "-a", "--delete", "--exclude", "lms", "--exclude", "rnn", "--exclude", "rb",
                            "--exclude", "bpbc", f"{pc}/build", *map(str, sorted(pc.glob("build_fix*"))), f"{pc}/compat",
                            f"{pc}/remote_build.sh", f"{ROOT}/src/openwatcom", f"{remote}:{base}/src/"], check=True)
            subprocess.run(["ssh", remote, f"cd ~/{base}/src && sh remote_build.sh"], check=True)
            subprocess.run(["rsync", "-a", "--delete", f"{root}/", f"{remote}:{base}/jobs/"], check=True)
            subprocess.run(["ssh", remote, f"cd ~/{base}/jobs && BIN=~/{base}/src; {link}"], check=True)
            subprocess.run(["rsync", "-a", f"{remote}:{base}/jobs/", f"{root}/"], check=True)
        else:
            subprocess.run(["sh", "-c", f"cd '{root}' && BIN='{ROOT / 'src/clang'}'; {link}"], check=True)
        print(f"C: {len(cjobs)} jobs in {time.time()-t0:.0f} s on {remote or 'localhost'}", flush=True)
        return {key: read_results(root / names[k], prog) for k, (key, impl, prog, p, runs) in enumerate(cjobs)}


def run_py(prog, p, runs):
    py = ROOT / "src" / "anneq_orig.py"
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        write_job(d, prog, p, runs)
        env = dict(os.environ, EQ_SEED=str(SEED))
        try:
            subprocess.run([sys.executable, str(py), prog, "TEST"], cwd=d, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3600)
        except subprocess.TimeoutExpired:
            pass
        return read_results(d, prog)


def run_numpy(p, trials, fix):
    from anneq_numpy import run_lms
    import warnings
    warnings.simplefilter("ignore")
    curve, ber, fail = run_lms(p["ch"], p["taps"], p["eta"], p["snr"], p["delay"], trials=trials,
                               bias=fix, seed=SEED)
    return dict(ber=ber, fail=fail, curve=list(curve), hung=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="clang,clang_fix,pyline,numpy,numpy_fix")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--py-trials-rnn", type=int, default=20, help="pure-Python RTRL is slow")
    a = ap.parse_args()
    impls = a.impls.split(",")
    jobs = [j for j in JOBS if a.only in j[0]]
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    results = {}

    cjobs = []
    for scen, prog, lab, p in jobs:
        if "clang" in impls and prog != "BPBC":
            cjobs.append(((scen, "clang", lab), "clang", prog, p, TRIALS))
        if "clang_fix" in impls and prog in ("LMS", "RNN", "BPBC"):
            cjobs.append(((scen, "clang_fix", lab), "clang_fix", prog, p, TRIALS if prog != "BPBC" else 20))
        for tag in VARIANTS.get(prog, ()):
            if f"clang_fix_{tag}" in impls:
                cjobs.append(((scen, f"clang_fix_{tag}", lab), f"clang_fix_{tag}", prog, p, TRIALS))
        if "pyline" in impls and prog == "RNN" and "clang" in impls:   # same-trial check for pyline RNN
            cjobs.append(((scen, f"clang_{a.py_trials_rnn}", lab), "clang", prog, p, a.py_trials_rnn))
    if cjobs:
        for key, r in run_c_jobs(cjobs, a.jobs).items():
            results[key] = (r, next(c[4] for c in cjobs if c[0] == key))
    hung = {(k[0], k[2]) for k, (r, _) in results.items() if k[1] == "clang" and r["hung"]}

    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {}
        for scen, prog, lab, p in jobs:
            if "pyline" in impls and prog in ("LMS", "RNN", "RB") and (scen, lab) not in hung:
                n = a.py_trials_rnn if prog == "RNN" else TRIALS
                futs[ex.submit(run_py, prog, p, n)] = ((scen, "pyline", lab), n)
            if prog == "LMS":
                for impl in ("numpy", "numpy_fix"):
                    if impl in impls:
                        futs[ex.submit(run_numpy, p, TRIALS, impl == "numpy_fix")] = ((scen, impl, lab), TRIALS)
        t0 = time.time()
        for f in cf.as_completed(futs):
            results[futs[f][0]] = (f.result(), futs[f][1])
        print(f"Python: {len(futs)} jobs in {time.time()-t0:.0f} s", flush=True)

    rows = {}
    for (scen, impl, lab), (r, n) in sorted(results.items()):
        c = r["curve"]
        rows.setdefault(scen, []).append(dict(
            impl=impl, setting=lab, trials=n, ber=r["ber"], fail_per_trial=r["fail"],
            final_err=(sum(c[-100:]) / 100 if len(c) >= 100 else None), hung=r["hung"]))
        if c:
            (OUT / "curves" / f"{scen}__{impl}__{lab}.txt").write_text("\n".join(f"{v:.7g}" for v in c) + "\n")
    for scen, rs in rows.items():   # merge: new rows replace same (impl, setting), others kept
        old = OUT / f"{scen}.csv"
        if old.exists():
            new = {(r["impl"], r["setting"]) for r in rs}
            rs = [r for r in csv.DictReader(open(old)) if (r["impl"], r["setting"]) not in new] + rs
        with open(OUT / f"{scen}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader(); w.writerows(rs)


if __name__ == "__main__":
    main()
