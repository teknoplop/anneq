"""Unfixed (results/rebuild, clang = pyline, 100 trials) vs fixed Python (results/fixed).
uv run --no-project python scenarios/compare_fixed.py > results/fixed/compare.md
"""
import csv, pathlib
D = pathlib.Path(__file__).resolve().parent.parent / "results"
UNFIX = {"bp_ch1": "clang_fix", "bp_ch2": "clang_fix"}   # BPBC: the BP-DEBUG build (faithful one crashes)


def load(d, scen):
    p = D / d / f"{scen}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(v, k="ber"):
    if v is None or v.get(k) in (None, "", "None"):
        return "hung" if v and v.get("hung") == "True" else "-"
    x = float(v[k])
    return f"{x:.3g}"


print("| scenario | setting | unfixed BER | fixed BER | fixed, 31-bit PN | unfixed retries/trial | fixed retries/trial | fixed gave up | unfixed final err | fixed final err |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted((D / "fixed").glob("*.csv")):
    scen = p.stem
    if "_eta_sweep" in scen:
        continue
    u = {r["setting"]: r for r in load("rebuild", scen) if r["impl"] == UNFIX.get(scen, "clang")}
    fx = {(r["impl"], r["setting"]): r for r in load("fixed", scen)}
    sets = sorted({s for _, s in fx}, key=lambda s: float(s.split("=")[1]))
    for s in sets:
        a, b, c = u.get(s), fx.get(("pyfix", s)), fx.get(("pyfix_pn31", s))
        print(f"| {scen} | {s} | {f(a)} | {f(b)} | {f(c)} | {f(a, 'fail_per_trial')} | {f(b, 'fail_per_trial')} | "
              f"{f(b, 'gaveup_per_trial')} | {f(a, 'final_err')} | {f(b, 'final_err')} |")
