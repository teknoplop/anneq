"""Byte-for-byte check: the C++ port (src/clang/anneq/) against the src/clang C builds and src/anneq_fixed.py.

uv run --no-project python src/tools/check_cpp.py        (or: make -C src/clang/anneq check)
Needs `make -C src/clang/anneq` (EQ_CPP_BUILD=build_san after `make -C src/clang/anneq sanitize` runs the ASan/UBSan
build instead) and `make -C src/clang all fixed variants`. Same configs, channels and
EQ_SEED as src/tools/check_pyport.py, whose helpers it reuses; the C references are compiled
again with -ffp-contract=off, as there (the C++ is built with it too). Every output file the
programs write must be byte-identical. Cases:
  1995      C++ EQ_NOFIX=ALL          vs src/clang/build/<p>          (LMS, RNN, RB; BPBC vs build_fix/bpbc:
                                         the faithful C BPBC segfaults, BP-DEBUG only removes screen output)
  TAG       C++ EQ_FIX=TAG            vs src/clang/build_fix_TAG/<p>  (LMS-FILL, LMS-BIAS, RNN-FFWD, RNN-KRON)
  all-C     C++ EQ_FIX=<all C tags>   vs src/clang/build_fix/<p>      (LMS, RNN, BPBC)
  TAG       C++ EQ_FIX=TAG            vs anneq_fixed.py EQ_FIX=TAG    (the Python-only fixes, no C counterpart)
  all       C++ (default: all fixes)  vs anneq_fixed.py               (LMS, RNN, RB, BPBC)
Exit status 1 on any difference.
"""
import os, pathlib, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_pyport as cp  # noqa: E402

CPP = cp.ROOT / "src/clang/anneq" / os.environ.get("EQ_CPP_BUILD", "build")   # build_debug, build_san: hardened / sanitised
ALL_C = "LMS-FILL,LMS-BIAS,RNN-FFWD,RNN-KRON,BP-DEBUG"
INPUTS = {"LMS_CON.TXT", "RNN_CON.TXT", "RB_CON.TXT", "WEIGHTS.TXT", "BP_CONST.TXT",
          "e:\\temp\\bp_const.txt", "e:\\temp\\weights.txt"}


def setup(d, prog, ch, rest, fix):
    cp.setup(d, prog, ch, rest)
    if prog == "BPBC" and ("BP-PATHS" in fix or fix == "ALL"):   # run-folder names (#17)
        (d / "e:\\temp\\bp_const.txt").rename(d / "BP_CONST.TXT")
        (d / "e:\\temp\\weights.txt").rename(d / "WEIGHTS.TXT")


def run(prog, kind, ref, fix, ch, rest):
    """kind: cpp | c | py. fix: EQ_FIX value, "ALL" (default env) or "" (1995: EQ_NOFIX=ALL for C++)."""
    d = pathlib.Path(tempfile.mkdtemp())
    setup(d, prog, ch, rest, fix)
    env = dict(os.environ, EQ_SEED=cp.SEED, MallocNanoZone="1")
    for k in ("EQ_FIX", "EQ_NOFIX", "EQ_MAX_ATTEMPTS", "EQ_RAND"):
        env.pop(k, None)
    if kind == "cpp":
        cmd = [str(CPP / prog.lower()), "TEST"]
        if fix == "":
            env["EQ_NOFIX"] = "ALL"
        elif fix != "ALL":
            env["EQ_FIX"] = fix
    elif kind == "c":
        (d / cp.EXE[prog]).symlink_to(ref)
        cmd = ["./" + cp.EXE[prog], "TEST"]
    else:
        if fix != "ALL":
            env["EQ_FIX"] = fix
        cmd = [sys.executable, str(cp.FIXED), prog, "TEST"]
    r = subprocess.run(cmd, cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    outs = {p.name: p.read_bytes() for p in d.iterdir() if p.name not in INPUTS and p.name != cp.EXE[prog]}
    return r.returncode, outs


def main():
    pc = cp.PC
    plan = [("1995", p, "c", pc / "build" / p.lower(), "") for p in ("LMS", "RNN", "RB")]
    plan.append(("1995", "BPBC", "c", pc / "build_fix" / "bpbc", ""))
    for tag, p in (("LMS-FILL", "LMS"), ("LMS-BIAS", "LMS"), ("RNN-FFWD", "RNN"), ("RNN-KRON", "RNN")):
        plan.append((tag, p, "c", pc / f"build_fix_{tag}" / p.lower(), tag))
    for p in ("LMS", "RNN", "BPBC"):
        plan.append(("all-C", p, "c", pc / "build_fix" / p.lower(), ALL_C))
    for tag, progs in (("RETRY-CAP", ("LMS", "RNN")), ("SEED-ONCE", ("LMS", "RNN", "RB", "BPBC")),
                       ("RAND-PREAMBLE", ("LMS", "RNN", "RB", "BPBC")), ("BP-PATHS", ("BPBC",))):
        plan += [(tag, p, "py", None, tag) for p in progs]
    plan += [("all", p, "py", None, "ALL") for p in ("LMS", "RNN", "RB", "BPBC")]

    tmp = pathlib.Path(tempfile.mkdtemp())
    bad = 0
    for what, prog, kind, cbin, fix in plan:
        ref = cp.nofma(cbin, tmp) if kind == "c" else None
        against = f"C {cbin.parent.name}" if kind == "c" else "anneq_fixed.py"
        for lab, ch, rest in cp.CASES[prog]:
            rc, mine = run(prog, "cpp", None, fix, ch, rest)
            _, theirs = run(prog, kind, ref, fix, ch, rest)
            ok = rc == 0 and mine == theirs and len(mine) > 0
            bad += not ok
            ber = next((v for k, v in mine.items() if k.lower().endswith("ber.txt")), b"?").decode().strip()
            note = "" if ok else f"  files {sorted(mine)} vs {sorted(theirs)}" if set(mine) != set(theirs) else \
                "  differs: " + ",".join(k for k in mine if mine[k] != theirs.get(k))
            print(f"{'OK  ' if ok else 'DIFF'} {what:13s} {prog:5s} {lab:16s} vs {against:20s} "
                  f"{len(mine)} files  BER {ber:>18s}{note}", flush=True)
    print("ALL BYTE-EXACT" if not bad else f"{bad} case(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
