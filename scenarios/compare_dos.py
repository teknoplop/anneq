"""Compare results/dos/dos.csv with the native clang runs in results/rebuild/<scenario>.csv.
uv run --no-project python scenarios/compare_dos.py  -> markdown table on stdout
BPBC is compared with clang_fix (the only native BPBC: faithful build crashes on a hosted OS)."""
import csv, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((ROOT / "results/dos/dos.csv").open()))
print("| scenario | setting | prog | DOS BER | native BER | DOS fail/trial | native fail/trial | final err (mean last 100) DOS / native | match | max curve diff (lines 2..) | DOS s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    impl = "clang_fix" if r["prog"] == "BPBC" else "clang"
    nat = {x["setting"]: x for x in csv.DictReader((ROOT / f"results/rebuild/{r['scenario']}.csv").open())
           if x["impl"] == impl}.get(r["setting"], {})
    def same(a, b):
        try: return abs(float(a) - float(b)) <= 1e-6 * max(1, abs(float(b)))
        except ValueError: return a == b
    def curve(i):
        q = ROOT / f"results/{'dos' if i == 'dos' else 'rebuild'}/curves/{r['scenario']}__{i}__{r['setting']}.txt"
        return [float(v) for v in q.read_text().split()] if q.exists() else []
    cd, cn = curve("dos"), curve(impl)
    # line 1 of ERRORS.TXT differs by design (fill_rand overrun clobbers errors[0] in DOS)
    cmax = max((abs(a - b) for a, b in zip(cd[1:], cn[1:])), default=None) if cd and cn else None
    r["final_err"] = f"{sum(cd[-100:]) / 100:.6f}" if len(cd) >= 100 else ""
    m = "yes" if r["ok"] == "True" and same(r["ber"], nat.get("ber", "")) and (cmax is None and not cd and not cn or cmax is not None and cmax < 1e-6) else "no"
    if r["ok"] != "True": m = "no result"
    print(f"| {r['scenario']} | {r['setting']} | {r['prog']} | {r['ber']} | {nat.get('ber','')} | {r['fail_per_trial']} | "
          f"{nat.get('fail_per_trial','')} | {r['final_err'][:8]} / {nat.get('final_err','')[:8]} | {m} | {'' if cmax is None else f'{cmax:.2g}'} | {r['secs']} |")
