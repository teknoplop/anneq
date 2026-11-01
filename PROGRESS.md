# Progress log (issue #9)

## Done
- src/borland/: 14 files extracted verbatim (src/tools/extract_1995.py).
- src/tools/prepare.py SRC OUT: rejoins wrapped #defines/strings -> lower-case files in OUT.
- Gemini bug identified (see results/RESULTS.md): ADALINE bias input never set
  (layer_0[nodes_0] stays 0 from calloc; shift_right_layer only shifts nodes_0 entries).
  Also: fill_rand writes nodes_0+2 weights into an nodes_0+1 array (heap overrun by one).
- DOS build of LMS works (src/openwatcom/borland.h force-included):
    export WATCOM=~/opt/watcom INCLUDE=~/opt/watcom/h PATH=~/opt/watcom/bino64:$PATH
    uv run --no-project python src/tools/prepare.py src/borland build/prep; cp build/prep/lms_init.h build/prep/lms_init.c
    cd build/prep; wcl -q -bt=dos -ml -fpi87 -fi=../../src/openwatcom/borland.h -i=../../src/openwatcom lms.c lms_init.c eq_funcs.c -fe=LMS.EXE
- DOSBox-X headless works (~5 s/trial incl. startup at default cycles):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy dosbox-x -silent -nogui -c "mount c ." -c "c:" -c "LMS.EXE TEST > OUT.TXT" -c exit
  ch1 7 taps eta .01 10 dB 5 runs -> BER 3.44e-3 (report ~3.5e-3).

## Next
- src/clang (clang), Python line port (now src/anneq_orig.py), numpy port (src/anneq_numpy.py), scenarios driver, RESULTS.md, other nets.

## Milestone 2 (session 2)
- src/openwatcom/build.sh builds LMS/RNN/RB/BPBC.EXE into build/dos (Watcom). eqcompat.[ch]: Borland rand + EQ_SEED.
- src/clang/makefile: `make all fixed` -> build/{lms,rnn,rb,bpbc}, build_fix/{lms,rnn}. Binaries must be run as
  ./LMS.EXE / ./RNN.EXE / ./RB.EXE (path is derived by stripping 7/6 chars from argv[0]).
  Never cp over an existing signed binary on macOS (SIGKILL) - rm first.
- src/anneq_orig.py (was py/line_port/{eq_funcs,lms,rnn,rb}.py): bit-exact with clang (same EQ_SEED). src/anneq_numpy.py.
- src/tools/apply_fixes.py: LMS-BIAS, RNN-FFWD, RNN-KRON fixes.
- scenarios/run_scenarios.py: all jobs -> results/rebuild/*.csv. Full run incl DOS:
    uv run --no-project --with numpy --with matplotlib python scenarios/run_scenarios.py > results/rebuild/run.log
- DOS vs clang LMS: bit-identical except ERRORS.TXT line 1 (fill_rand overrun clobbers errors[0] in DOS).
## Next: plots (scenarios/plot.py), RESULTS.md, BPBC one-off run, verdict.

## Milestone 3
- BPBC: faithful build segfaults on hosted OS (debug print_weights w/ swapped args, NULL weights)
  -> fix BP-DEBUG in src/tools/apply_fixes.py; run src/clang/build_fix/bpbc (files are literally named
  'e:\temp\bp_const.txt' etc. in cwd on macOS). BP_CONST: m_taps bits epochs snr runs n0 n1 n2 lr mom.
- BLOCKER: Rosetta x86_64 processes now hang in state U (wcl stuck, unkillable) -> cannot rebuild DOS
  EXEs this session. build/dos/*.EXE from 15:11 are fine for LMS/RNN/RB; DOS BPBC not run (would wait
  for a key: needs kbhit shim added to src/openwatcom/borland.h after the 15:11 build). Reboot fixes.
- Full run: run_scenarios.py --dos-trials 20 --dos-trials-rnn 20 (DOS at 100 trials ~1.5 min/job, too slow).
- Endless retrain: LMS.C/RNN.C `while(fail)` has no cap. fig5_4 eta=0.0001 and fig9_3_ada 6 dB never
  finish (also in DOS) -> driver runs native first with timeout 60 s, marks hung=True, skips those in DOS.
- void main => garbage exit status; driver ignores it.
- scenarios/extra_bp_online.py(+.txt): per-sample backprop re-creation converges 20/20 (report: fails 1 in 5).
- Running: full driver (bg). Then: plot.py, RESULTS.md.

## Plan change (coordinator): DOS parked, native arm64 + Python only
- DOS TO REVISIT AFTER A REBOOT (Rosetta hang): rebuild with src/openwatcom/build.sh (now has the kbhit/getch
  shim), run BPBC under DOSBox (mount E: with TEMP\), full 100-trial DOS runs. Partial DOS results kept in
  scenarios/dos_partial/ (fig5_3 + fig5_4 eta .1/.01/.001, at 100 and 20 trials, EQ_SEED=1000).
- src/clang binaries confirmed `Mach-O 64-bit executable arm64`; no Rosetta used.
- Driver rewritten: parallel (xargs -P for C, ProcessPool for Python). REMOTE=mini.local runs C jobs on the
  Mac mini in ~/anneq-runs/ (rsync prepared sources, src/clang/remote_build.sh builds there with clang, xargs -P10,
  rsync back). Mini rules: only ~/anneq-runs/, no USB/FPGA tools; cap -P 6 if asked.
    REMOTE=mini.local uv run --no-project --with numpy --with matplotlib python scenarios/run_scenarios.py --jobs 10
- src/clang/sanitize.sh [build|build_fix] -> results/sanitize.txt (ASan+UBSan fault list).
- src/tools/apply_fixes.py TAGS -> src/clang/build_fix_<TAG> one-fix variants (LMS-FILL, LMS-BIAS, RNN-FFWD, RNN-KRON);
  driver impls clang_fix_<TAG>.

## Milestone 4 (done)
- Full native run on mini (147 C jobs, 184 s) + local Python (141 jobs). Per-fix variants run.
- scenarios/plot.py -> results/rebuild/*.png (incl. overview_telephone.png). results/RESULTS.md written
  (comparison tables, 12 bugs, sanitizer table, memory-safe choice = Python line port w/ EQ_STRICT, verdict).
- Python line port (src/anneq_orig.py): EQ_STRICT=1 raises on the fill_rand overrun.
## Still open
- DOS after reboot (see above). BPBC learning rate/momentum unknown (0.1/0.5 assumed).
- Toy-channel RNN/RBF figures (7.3, 8.3) not reproduced (report ~3-10x worse); cause unknown.
- numpy port covers ADALINE only; py line port has no BPBC.

## Milestone 5: DOS runs (#9, branch dos-runs)
- Rosetta works again: src/openwatcom/build.sh rebuilt build/dos/*.EXE locally (kbhit shim in).
- dosbox-x installed on the mini (brew). scenarios/run_dos.py: job dirs (C: = exe+inputs, E:\TEMP\ for
  BPBC), cycles=max, core=dynamic, EQ_SEED=1000, xargs -P6 on the mini in ~/anneq-runs/dos/.
  Output results/dos/dos.csv (+curves/). Merges by (scenario,setting), so --only reruns are safe.
    REMOTE=mini.local uv run --no-project python scenarios/run_dos.py --jobs 6
- BPBC faithful DOS build runs (no crash, unlike hosted OS): 2-trial smoke test ok, ~7 s/job.
- Running: full 100-trial (BPBC 20) DOS run on the mini.
- DONE: full DOS run, 82 jobs in 64 min (5 x fig9_3_ada eta=.08 hit the 1 h timeout; now in SKIP).
  LMS/RNN/RB 72/72 match clang (BER + curves from line 2). BPBC faithful runs in DOS; ch1 matches the
  native BP-DEBUG fix at 10/14/21 dB, and small x87-vs-arm64 FP drift elsewhere. Written up in RESULTS.md "DOS runs".
  scenarios/compare_dos.py -> results/dos/compare.md. dos_partial/ superseded by results/dos/.
- DOS item of #9 finished.

## FPGA milestone (#22, branch fpga) - done
- gateware/ (was fpga/): Icepi Zero (ECP5-25F) batch-backprop engine for BPBC 2-6-6-1. rtl/bp_engine.v (1 trial/engine,
  Q3.14/Q3.28 fixed point, tanh LUT), rtl/top.v (10 engines, UART 2 Mbaud on K15/K16, 50 MHz clk M1),
  ref/bpfix.c (bit-exact C model + double/fixed timing), sim/tb_engine.cpp (verilator), py/anneq_fpga.py
  (data/jobs, numpy trainer, BER, board protocol). Pinout from vendor/icepi-zero (submodule).
- Mini: tools from brew already present (yosys, nextpnr-ecp5, ecppack, openFPGALoader, verilator; no iverilog).
  Work dir ~/anneq-runs/fpga/{src,work,native}. gateware/run_mini.sh rebuilds, verifies and reruns everything.
  SRAM loads only. Board appears as FT231X /dev/cu.usbserial-*.
- Results in gateware/RESULTS.md and results/fpga/: sim and board bit-exact with C model; 10 engines 90% LUT, 54 MHz Fmax;
  2.15e6 presentations/s on chip, 1.41e6 end to end; M4 1 core C 7.3-25e6/s, numpy 3.7e6/s; ~500x a 486.
- Possible follow-ups (not needed for the verdict): share tanh LUT between engine pairs, slimmer register files
  to fit ~14 engines; PLL clock.

## Welstead book disk (#21)
- Disk unpacked via its own installer in DOSBox-X (INT 16h key-feeder TSR); compared with src/borland/.
  All 14 files independent (longest shared token run 14 = loop headers); no bug in #10-#19 inherited.
  Write-up: results/WELSTEAD.md. Disk contents stay git-excluded.

## Python fixes milestone (#10-#18, branch pyfix)
- Fixed variant, now src/anneq_fixed.py: all fixes by default (was EQ_FIXED=1), or EQ_FIX=TAG,.. / EQ_NOFIX=TAG. C-counterpart tags
  LMS-FILL LMS-BIAS RNN-FFWD RNN-KRON BP-DEBUG; Python-only RETRY-CAP (10 attempts, GAVEUP.TXT, ERRORS_ALL.TXT)
  SEED-ONCE RAND-PREAMBLE BP-PATHS MINOR. New BPBC port (now in src/anneq_orig.py / anneq_fixed.py).
- Port fixes found by the check: noise() and BPBC's 0.3*URAND now evaluate the unbracketed C URAND macro as C does.
- src/tools/check_pyport.py (needs `make -C src/clang all fixed variants`): 28 cases byte-identical vs clang refs built with
  -ffp-contract=off (default src/clang builds use FMA; only BPBC's 15-digit output notices). Log results/check_pyport.txt.
- scenarios/run_fixed.py: same jobs as run_scenarios.py -> results/fixed/ (impls pyfix, pyfix_pn31).
    REMOTE=mini.local uv run --no-project python scenarios/run_fixed.py --jobs 8     (168 jobs, 115 s on the M4)
    ... --sweep   (RNN eta sweep + telephone ADALINE sweep; chosen RNN eta 0.12 toy / 0.08 telephone)
  Mini work dir ~/anneq-runs/pyfix/. Keep logs outside results/fixed/ while running (the rsync back overwrites it).
  scenarios/compare_fixed.py -> results/fixed/compare.md; scenarios/plot.py --out fixed.
- results/RESULTS_FIXED.md written: per-figure unfixed vs fixed, list of report statements that change
  (main one: Fig 9.3 reversed above 10 dB, RNN beats ADALINE; §7.3/§10 sensitivity and "second best" claims).
- Issues #10-#18 commented with the commit. Not closed.
- Possible follow-ups: same fixes in a modern C source (checklists of #12, #13, #17, #18 ask for C); telephone
  RBF run with the random preamble; BPBC decision delay for channel 2.

## 2026 re-run (unprefixed modules in src/, issue #20; was y2026/PROGRESS.md)


Resume: `uv run --extra test pytest -q` (repo root), then `uv run scenarios/run_y2026.py all --jobs 8`,
then `uv run scenarios/plot_y2026.py`. Remote optional: `Y2026_HOST=mini.local` (needs uv there).

- [x] M1 framework: uv project, channels/data/metrics/linear/trellis/neural/experiments; 18 tests pass.
  - Fixed during build: first LFSR was wrong (stuck); now Galois form, m-sequence property tested.
- [x] M2 smoke runs of both scenarios (all methods produce sane BERs).
- [x] M3 full sweeps done (offline 648 s, preamble 696 s on 8 local workers; Mac mini not used).
  - Bug found: RLS covariance drifted on 100k-symbol offline runs (RLS/DFE BER ~0.3). Fixed by symmetrising P; regression test added; offline and preamble re-run.
- [x] M4 figures (results/y2026/figures/), results/y2026/tables.md, docs/report_2026.md written. Remaining: review; remote path untested; NN high-SNR undertraining noted as limitation.
- Note: the Mac mini came back online after all sweeps had already finished locally. No sweeps were
  left to move, so the mini was not used and uv was not installed there. No local sweep is running.
  Any future sweep: `Y2026_HOST=mini.local uv run scenarios/run_y2026.py ...` (runs in ~/anneq-runs/y2026/,
  needs uv on the mini first). Heavy sweeps go to the mini (M4, 10 cores) with `--jobs 10`;
  keep this MacBook (M1) at `--jobs 2`-`4` or run nothing here.

## C++ port (issue #25)
- src/clang/anneq/: eq_funcs.hpp (Channel, Inputs = [taps, bias, feedback] with gather(), Layer with a
  [node][input] weight Matrix, Borland rand/EQ_SEED, noise, PN/random bits, config and output
  files, EQ_FIX/EQ_NOFIX tags) + lms/rnn/rb/bpbc.cpp, same flow and files as 1995. C++23 (branch
  cpp23): each .cpp mirrors its Borland *_INIT.H + main .C block for block (same globals, init
  function, comment headings, names and statement order) and eq_funcs.hpp mirrors EQ_FUNCS.H/.C
  (ANN_STR / NODE_STR / WEIGHT_STR as vectors of structs, functions in EQ_FUNCS.C order). Swaps:
  printf -> std::print/println (format specs reproduce the printf output; NaN written as "nan"
  like macOS printf), calloc -> std::vector/std::array, index loops that only reach an element ->
  range-for, views::zip (and zip with iota for enumerate), ranges::fold_left in dist(),
  std::expected for reading the constants files, enum class Fix + std::to_underlying, constexpr
  for the macros, std::mdspan (Grid<3>, sense[k, l, j]) for the RNN sensitivities. Index loops
  kept where the index means something (RTRL k/l/j/i, shifts, derivatives). Builds with Apple
  clang 21 and Homebrew clang 23 (views::enumerate skipped: not in Apple libc++). Allman braces
  via the repo-root .clang-format. clang++, -ffp-contract=off. All fixes by default; EQ_NOFIX=ALL gives the 1995 behaviour.
- `make -C src/clang/anneq` (build/), `debug` (libc++ hardening), `sanitize` (ASan+UBSan), `check` (runs all three builds).
- src/tools/check_cpp.py: 62 cases (1995, each C tag, all-C vs the -ffp-contract=off C builds;
  Python-only tags and all fixes vs anneq_fixed.py), all byte-exact (results/check_cpp.txt).
  Same 62 cases byte-exact with EQ_CPP_BUILD=build_debug and build_san, no sanitizer reports.

## Pythonic simulators (src/anneq.py, issue #24, branch pythonic)

- src/anneq.py: the four simulators (lms, rnn, rbf, bp) as idiomatic Python 3.15: dataclass configs,
  numpy Generator with SeedSequence per trial, vectorised where the algorithm allows (channel, delay
  lines, RBF centres and test, batch backprop), per-sample loops for LMS and RTRL. Same algorithms as
  anneq_fixed.py (all fixes, random preamble by default, retry cap 10). Library vs own code is listed in
  the module docstring (padasip and sklearn MLPRegressor considered and not used).
- Tests: `uv run --python 3.15 --no-project --with numpy --with scipy --with pytest pytest -q src/tests`
  (8 tests, incl. RTRL sensitivities and backprop gradients against finite differences).
- Comparison: `REMOTE=mini.local uv run --python 3.15 --no-project --with numpy --with scipy python
  scenarios/compare_pythonic.py --jobs 8` (84 points, ~10 s on the M4 in ~/anneq-runs/pythonic/)
  -> results/pythonic/compare.md. 79/84 within |z| < 3; the five outside (bp_ch1 10/14/21 dB, fig5_3
  taps=3 and extra_ada10_real 6 dB final error) are explained there by seed-to-seed spread, checked by
  rerunning both implementations with other seeds.
