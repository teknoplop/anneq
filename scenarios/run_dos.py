"""Run the report scenarios on the 16-bit DOS builds (build/dos/*.EXE) under headless DOSBox-X.

uv run --no-project python scenarios/run_dos.py [--only SUBSTR] [--jobs 6] [--trials 100] [--bp-trials 20]
  REMOTE=host   run on that host in ~/anneq-runs/dos/ (needs dosbox-x there); default: localhost
Builds one job dir per scenario point: C: = program + inputs, E:\\TEMP\\ for BPBC's hard-coded
paths. DOSBox-X runs at cycles=max with EQ_SEED set, so results compare bit for bit with clang.
Jobs that never finish natively (endless retrain, see PROGRESS.md) are skipped.
Output: results/dos/dos.csv and results/dos/curves/<scenario>__dos__<setting>.txt
"""
import argparse, csv, os, pathlib, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scenarios"))
from run_scenarios import JOBS, EXE, SEED, con_line  # noqa: E402

OUT = ROOT / "results" / "dos"
SKIP = {("fig5_4", "eta=0.0001")} | {("fig9_3_ada", f"snr={s}") for s in (6, 10, 14, 16, 18, 20)}  # diverge, never finish
TIMEOUT = 3600
CONF = """[sdl]
output=surface
[dosbox]
memsize=16
[cpu]
core=dynamic
cycles=max
[autoexec]
mount c .
mount e e
c:
set EQ_SEED={seed}
{exe} TEST > OUT.TXT
exit
"""
RUNNER = """#!/bin/sh
cd "$1" || exit 1
t0=$(date +%s)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy perl -e 'alarm {timeout}; exec @ARGV' \\
  dosbox-x -silent -nogui -exit -conf dos.conf >dosbox.log 2>&1
echo $(( $(date +%s) - t0 )) > SECS
exit 0
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--bp-trials", type=int, default=20)
    a = ap.parse_args()
    remote = os.environ.get("REMOTE")
    jobs = [j for j in JOBS if a.only in j[0] and (j[0], j[2]) not in SKIP]
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td) / "jobs"
        root.mkdir()
        (root / "run.sh").write_text(RUNNER.format(timeout=TIMEOUT))
        os.chmod(root / "run.sh", 0o755)
        names = []
        for k, (scen, prog, lab, p) in enumerate(jobs):
            d = root / f"D{k:03d}"
            (d / "e" / "TEMP").mkdir(parents=True)
            runs = a.bp_trials if prog == "BPBC" else a.trials
            w = "".join(f"{x}\n" for x in p["ch"])
            if prog == "BPBC":
                (d / "e" / "TEMP" / "BP_CONST.TXT").write_text(con_line(prog, p, runs))
                (d / "e" / "TEMP" / "WEIGHTS.TXT").write_text(w)
            else:
                (d / f"{prog}_CON.TXT").write_text(con_line(prog, p, runs))
                (d / "WEIGHTS.TXT").write_text(w)
            shutil.copy(ROOT / "build" / "dos" / EXE[prog], d / EXE[prog])
            (d / "dos.conf").write_text(CONF.format(seed=SEED, exe=EXE[prog]))
            names.append(d.name)
        # longest first: RNN / telephone channel jobs dominate
        order = sorted(range(len(jobs)), key=lambda k: (jobs[k][1] not in ("RNN", "BPBC"), k))
        (root / "list.txt").write_text("\n".join(names[k] for k in order) + "\n")
        cmd = f"cat list.txt | xargs -P {a.jobs} -n 1 ./run.sh"
        t0 = time.time()
        if remote:
            base = "anneq-runs/dos"
            subprocess.run(["ssh", remote, f"mkdir -p ~/{base}"], check=True)
            subprocess.run(["rsync", "-a", "--delete", f"{root}/", f"{remote}:{base}/jobs/"], check=True)
            subprocess.run(["ssh", remote, f"cd ~/{base}/jobs && PATH=/opt/homebrew/bin:$PATH; {cmd}"], check=True)
            subprocess.run(["rsync", "-a", f"{remote}:{base}/jobs/", f"{root}/"], check=True)
        else:
            subprocess.run(["sh", "-c", f"cd '{root}' && {cmd}"], check=True)
        wall = time.time() - t0
        (OUT / "curves").mkdir(parents=True, exist_ok=True)
        rows = []
        for k, (scen, prog, lab, p) in enumerate(jobs):
            d = root / names[k]
            def f(*parts):
                q = d.joinpath(*parts)
                return [float(v) for v in q.read_text().split()] if q.exists() else []
            if prog == "BPBC":
                ber, fail, curve = f("e", "TEMP", "BER.TXT"), [], f("e", "TEMP", "ERRORS.TXT")
            else:
                ber, fail, curve = f("BER.TXT"), f("FAIL.TXT"), f("ERRORS.TXT")
            secs = (d / "SECS").read_text().strip() if (d / "SECS").exists() else ""
            if curve:
                (OUT / "curves" / f"{scen}__dos__{lab}.txt").write_text("".join(f"{v}\n" for v in curve))
            rows.append(dict(scenario=scen, setting=lab, impl="dos", prog=prog,
                             trials=a.bp_trials if prog == "BPBC" else a.trials,
                             ber=ber[0] if ber else "", fail_per_trial=fail[0] if fail else "",
                             final_err=curve[-1] if curve else "", errors_line1=curve[0] if curve else "",
                             secs=secs, ok=bool(ber)))
        new = {(r["scenario"], r["setting"]) for r in rows}
        csvp = OUT / "dos.csv"
        old = [r for r in csv.DictReader(csvp.open())] if csvp.exists() else []
        rows = [r for r in old if (r["scenario"], r["setting"]) not in new] + rows
        with csvp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"DOS: {len(jobs)} jobs in {wall:.0f} s on {remote or 'localhost'} "
              f"({sum(1 for r in rows if r['ok'] in (True, 'True'))} ok of {len(rows)} in csv)", flush=True)


if __name__ == "__main__":
    main()
