"""Bit-exact check: src/anneq_orig.py and src/anneq_fixed.py against the src/clang clang builds (same EQ_SEED).

uv run --no-project python src/tools/check_pyport.py
Needs `make -C src/clang all fixed variants` (prepared sources + default binaries). Every output
file (ERRORS/BER/FAIL) must be byte-identical to a reference binary compiled here from the same
prepared sources with -ffp-contract=off. The src/clang default builds let clang fuse a*b+c into one
FMA instruction on arm64 (C allows it; Python, the 1995 x87 code and the C abstract machine do not
fuse). That never changes the LMS/RNN/RB output files, but BPBC prints 15 digits after 1000 batch
epochs, so there the default build differs in the last digits; the comparison with the default
binaries is printed too, for information. Cases:
  unfixed   src/clang/build/<prog>       vs anneq_orig.py           (LMS, RNN, RB)
            src/clang/build_fix/bpbc     vs anneq_orig.py BPBC     (the faithful C BPBC segfaults on a
                                         hosted OS; BP-DEBUG only removes screen output)
  TAG       src/clang/build_fix_TAG/<p>  vs anneq_fixed.py EQ_FIX=TAG (LMS-FILL, LMS-BIAS, RNN-FFWD, RNN-KRON)
  all-C     src/clang/build_fix/<p>      vs anneq_fixed.py EQ_FIX=<all C tags> (LMS, RNN, BPBC)
Exit status 1 on any difference.
"""
import os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PC = ROOT / "src/clang"
ORIG, FIXED = ROOT / "src" / "anneq_orig.py", ROOT / "src" / "anneq_fixed.py"
SEED = "1000"
CH1, CH2 = [1.0, 0.5], [0.5, 1.0]
REAL = [0.047, 0.321, 0.783, 0.427, 0.293, 0.100, 0.041, 0.038, 0.020, 0.011]

# (prog, label, channel, con-line fields after "m bits")
CASES = {
    "LMS": [("ch1 7tap 10dB", CH1, "10 4 0 7 0.01"), ("ch2 delay7 6dB", CH2, "6 3 7 7 0.01"),
            ("eta=0.09 retries", CH1, "21 3 0 7 0.09"), ("telephone", REAL, "14 2 5 15 0.008")],
    "RNN": [("ch1 2-3", CH1, "10 3 0 2 3 0.005"), ("telephone 10-4", REAL, "16 1 5 10 4 0.003")],
    "RB": [("ch1", CH1, "10 3 2 8 0.125"), ("ch2", CH2, "14 3 2 8 0.125")],
    "BPBC": [("ch1 6dB", CH1, None)],
}
OUTS = {"LMS": ["ERRORS.TXT", "BER.TXT", "FAIL.TXT"], "RNN": ["ERRORS.TXT", "BER.TXT", "FAIL.TXT"],
        "RB": ["BER.TXT"], "BPBC": ["e:\\temp\\errors.txt", "e:\\temp\\ber.txt"]}
EXE = {"LMS": "LMS.EXE", "RNN": "RNN.EXE", "RB": "RB.EXE", "BPBC": "BPBC.EXE"}


def setup(d, prog, ch, rest):
    w = "".join(f"{x}\n" for x in ch)
    if prog == "BPBC":   # m bits epochs snr runs n0 n1 n2 lr mom
        (d / "e:\\temp\\bp_const.txt").write_text(f"{len(ch)} 40 200 6 2 2 6 6 0.1 0.5\n")
        (d / "e:\\temp\\weights.txt").write_text(w)
    else:
        (d / f"{prog}_CON.TXT").write_text(f"{len(ch)} 1000 {rest}\n")
        (d / "WEIGHTS.TXT").write_text(w)


def nofma(cbin, tmp):
    """Compile the prepared sources next to cbin again with -ffp-contract=off."""
    src = cbin.parent
    prog = cbin.name
    out = tmp / f"{src.name}_{prog}"
    if not out.exists():
        init = "bp_init.h" if prog == "bpbc" else f"{prog}_init.h"
        extra = ["-include", str(ROOT / "src/openwatcom" / "rb_missing.h")] if prog == "rb" else []
        subprocess.run(["clang", "-O2", "-std=gnu89", "-w", "-Dfar=", "-ffp-contract=off", f"-I{PC}/compat",
                        f"-I{ROOT}/src/openwatcom", *extra, f"-I{src}", "-o", str(out), str(src / f"{prog}.c"),
                        "-x", "c", str(src / init), "-x", "none", str(src / "eq_funcs.c"),
                        str(ROOT / "src/openwatcom" / "eqcompat.c"), "-lm"], check=True)
    return out


def run_pair(prog, cbins, fix, ch, rest):
    out = []
    for kind in ("c", "c_fma", "py"):
        d = pathlib.Path(tempfile.mkdtemp())
        setup(d, prog, ch, rest)
        env = dict(os.environ, EQ_SEED=SEED, MallocNanoZone="1")
        env.pop("EQ_FIX", None); env.pop("EQ_NOFIX", None)
        if kind != "py":
            (d / EXE[prog]).symlink_to(cbins[kind])
            cmd = ["./" + EXE[prog], "TEST"]
        else:
            if fix:
                env["EQ_FIX"] = fix
            cmd = [sys.executable, str(FIXED if fix else ORIG), prog, "TEST"]
        subprocess.run(cmd, cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        out.append({n: (d / n).read_bytes() if (d / n).exists() else None for n in OUTS[prog]})
    return out


def main():
    allc = "LMS-FILL,LMS-BIAS,RNN-FFWD,RNN-KRON,BP-DEBUG"
    plan = []
    for prog in ("LMS", "RNN", "RB"):
        plan.append(("unfixed", prog, PC / "build" / prog.lower(), ""))
    plan.append(("unfixed", "BPBC", PC / "build_fix" / "bpbc", ""))
    for tag, prog in (("LMS-FILL", "LMS"), ("LMS-BIAS", "LMS"), ("RNN-FFWD", "RNN"), ("RNN-KRON", "RNN")):
        plan.append((tag, prog, PC / f"build_fix_{tag}" / prog.lower(), tag))
    for prog in ("LMS", "RNN", "BPBC"):
        plan.append(("all-C", prog, PC / "build_fix" / prog.lower(), allc))
    bad = 0
    tmp = pathlib.Path(tempfile.mkdtemp())
    for what, prog, cbin, fix in plan:
        cbins = {"c": nofma(cbin, tmp), "c_fma": cbin}
        for lab, ch, rest in CASES[prog]:
            c, cf, p = run_pair(prog, cbins, fix, ch, rest)
            ok = c == p and all(v is not None for v in c.values())
            bad += not ok
            ber = (c.get("BER.TXT") or c.get("e:\\temp\\ber.txt") or b"?").decode().strip()
            fma = "same" if cf == p else "differs (FMA)"
            print(f"{'OK  ' if ok else 'DIFF'} {what:9s} {prog:5s} {lab:16s} BER {ber:>18s}  default build: {fma}",
                  flush=True)
    print("ALL BIT-EXACT" if not bad else f"{bad} case(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
