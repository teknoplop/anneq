# Rebuilding the 1995 equaliser simulations: results

Tracking issue: teknoplop/anneq#9. All numbers below come from `results/rebuild/*.csv`
(regenerate with `scenarios/run_scenarios.py`, then `scenarios/plot.py` for the PNGs).
"Report" values are read off the 1995 figures, so they are approximate.
The same scenarios rerun with the bugs below fixed: `RESULTS_FIXED.md` (`fixed/`).

## What was built and where it ran

| Implementation | What it is | Where | Trials |
|---|---|---|---|
| `src/borland/` | the listings, extracted verbatim from the report transcription | — | — |
| **dos** | Open Watcom 2.0, 16-bit DOS large model, `src/openwatcom/` shim, DOSBox-X headless | this Mac (x86 compiler under Rosetta) | partial only, see below |
| **clang** | `src/clang/`: the same sources, arm64 clang, conio shim, `-Dfar=` | Mac mini (`REMOTE=mini.local`), 10 parallel jobs | 100 |
| **clang_fix**, **clang_fix_TAG** | the same build with all fixes, or one fix (`src/tools/apply_fixes.py`) | Mac mini | 100 (BPBC 20) |
| **pyline** | `src/anneq_orig.py`: line-for-line Python, stdlib only | this Mac, parallel | 100 (RNN 20, pure-Python RTRL is slow) |
| **numpy** / **numpy_fix** | `src/anneq_numpy.py`: idiomatic numpy ADALINE, all trials at once (fix = bias input on) | this Mac | 100 |

Only the source changes needed to build are made, all outside `src/borland/`: rejoin print-wrapped
lines (`src/tools/prepare.py`), one semicolon the scan hides (BP_INIT.H, marked `/*?*/`), the missing
`int bit_no;` in RB.C (`src/openwatcom/rb_missing.h`), and no-op screen calls. Every build uses Borland
C++ 3.1's `rand()` (`src/openwatcom/eqcompat.c`) and, with `EQ_SEED`, a fixed seed for the k-th
`srand()` call instead of the clock.

**Agreement between implementations.** With the same seeds, clang and the Python line port give
identical BER, failure counts and error curves in every scenario (all four programs). The DOS
build matches clang exactly too, on every point that was run in DOS (e.g. Fig 5.3, 7 taps: final
error 0.011716576 in both; Fig 5.4, η=0.1: BER 0.042236 and 11.7 retrains per trial in both). The one
difference is line 1 of `ERRORS.TXT`, which the DOS program corrupts (bug 1). numpy uses a
different random generator, and its numbers agree with the others to within trial-to-trial
scatter (e.g. Fig 5.5 at 10 dB: 3.42e-3 C, 3.48e-3 numpy).

**DOS status.** Two DOS runs got through Fig 5.3 and Fig 5.4 (η=0.1/0.01/0.001), at 100 and 20
trials (`scenarios/dos_partial/`). After that, Rosetta x86 processes started hanging in an
unkillable state, so the Watcom build could not be redone with the key-press shim BPBC needs. As
instructed, DOS is parked until after a reboot (see PROGRESS.md). At default DOSBox speed one
7-tap trial takes about 4 s, about 1.5 min per 100-trial job at `cycles=max`.

## Original vs rebuilt (BER, 100 trials, C = clang = pyline = DOS)

ADALINE, channel 1, 7 taps, η=0.01 (Fig 5.5):

| SNR dB | report | C | numpy | C bias-fixed |
|---|---|---|---|---|
| 6 | ~4e-2 | 4.24e-2 | 4.24e-2 | 4.29e-2 |
| 10 | ~3.5e-3 | 3.42e-3 | 3.48e-3 | 3.52e-3 |
| 12 | ~3e-4 | 3.2e-4 | 3.7e-4 | 3.4e-4 |
| 14 | ~7e-6 | 8e-6 | 1.4e-5 | 7e-6 |

Channel 2 with delay 7 (Fig 5.7) gives the same values (4.19e-2, 3.19e-3, 2.95e-4, 8e-6), as the
report says. Fig 5.3: the final error falls with taps, 0.212 / 0.058 / 0.022 / 0.0117 / 0.0126
for 1/2/3/7/11 taps. That supports the report's "7 is enough" (it read about 0.015 for 7 taps).

Telephone channel (Fig 9.3), see `out/overview_telephone.png`:

| SNR dB | report ADALINE | ADALINE η=0.008, 15 taps | ADALINE 10 taps | report RNN | RNN as written | RNN feedback fixed |
|---|---|---|---|---|---|---|
| 6 | ~1.4e-1 | 1.45e-1 | 1.41e-1 | ~1.4e-1 | 1.38e-1 | 1.37e-1 |
| 10 | ~6e-2 | 6.18e-2 | 6.05e-2 | ~6e-2 | 6.41e-2 | 6.00e-2 |
| 14 | ~1.1e-2 | 1.10e-2 | 1.05e-2 | ~1.7e-2 | 1.73e-2 | 1.31e-2 |
| 16 | ~2.3e-3 | 2.30e-3 | 2.35e-3 | ~6.5e-3 | 6.57e-3 | 4.03e-3 |
| 18 | ~2.4e-4 | 2.55e-4 | 2.68e-4 | ~2e-3 | 1.78e-3 | 8.1e-4 |
| 20 | ~6e-6 | 9e-6 | 1.4e-5 | ~3.5e-4 | 3.23e-4 | 8.5e-5 |

Both curves in Fig 9.3 are reproduced closely, but only once the ADALINE learning rate is taken
as **0.008**. With the printed 0.08, LMS diverges in every trial and the program never finishes
(bug 5). The report's 0.08 is almost certainly a slip for 0.008.

Toy-channel RNN and RBF (Figs 7.3, 8.3) are **not** reproduced. The programs as listed make
about 3 to 10 times fewer errors than the report's figures:

| SNR dB | report RNN | C RNN | report RBF ch1 | C RBF ch1 | ADALINE 2 taps (extra) |
|---|---|---|---|---|---|
| 6 | ~1.1e-1 | 4.26e-2 | ~1.15e-1 | 3.66e-2 | 4.26e-2 |
| 10 | ~4.5e-2 | 6.24e-3 | ~4.5e-2 | 3.84e-3 | 6.09e-3 |
| 14 | ~8e-3 | 1.78e-4 | ~8.5e-3 | 5.8e-5 | 1.32e-4 |
| 20 | ~4e-5 | 0 | ~4e-5 | 0 | 0 |

The ADALINE and telephone-channel figures do match, so the noise generator and SNR scale are not
the cause. These two figures were probably made with an earlier version of the programs or with
other settings. The cause cannot be found from the listing. Note also that, with the same 2 inputs,
the ADALINE (last column) is no better than the RNN and slightly worse than the RBF.

Backprop (BPBC, batch 2-6-6-1, 40 samples, 1000 epochs; the learning rate 0.1 and momentum 0.5 are
assumptions because the report does not give them): channel 1 BER 6.8e-2 / 1.6e-2 / 8e-4 / 1e-5 at
6/10/14/21 dB, with training error near 1e-5. It over-fits 40 noisy samples. Channel 2 does not
train at all, because BPBC.C has no decision delay. `extra_bp_online.py` re-creates the per-sample
backprop the report abandoned, from its description (that program is not in the listing). It
converged in 20 of 20 trials at 40, 21 and 10 dB. The report says it failed in at least 1 trial in 5.

## Bugs and faults found

Fixes are in `src/tools/apply_fixes.py`. Each fix's effect is measured as `clang_fix_<TAG>` vs `clang`.

1. **LMS-FILL: heap overrun in LMS.C** (`fill_rand(..., nodes_0+nodes_1+1, ...)` on an array of
   nodes_0+1 weights). The argument was copied from RNN.C. Each trial writes one random weight past
   the end of the weight array. In DOS this lands in the next heap block, the averaged error
   record, so the **first point of every ADALINE ERRORS.TXT is wrong** (0.667 instead of 1.602 in
   Fig 5.3, 7 taps). On macOS it either corrupts that value or makes malloc abort at random, so
   `src/clang` gives every `calloc` one spare element. Effect on results otherwise: none (Fig 5.5 BER
   with the fix: 4.23e-2, 3.42e-3, 3.6e-4, 6e-6, the same within scatter).
2. **LMS-BIAS: the ADALINE has no bias — this is the bug Gemini found** (`git -C <main checkout> diff`:
   it widened `shift_right_layer` to nodes_0+1 and forced the bias input to 1.0). LMS.C never sets
   the bias input `layer_0[nodes_0]`. It stays 0 from `calloc`, so the bias weight is trained
   against a zero input and does nothing. Figure 5.1 of the report shows a −1.0 bias. RNN.C and
   BPBC.C set theirs and LMS.C does not. It has been there since 8 Apr 1995 ("remove biases from
   structures"). Effect: **none that matters**. With ±1 symbols and zero-mean noise the best bias is
   0, so Fig 5.5 with the fix gives 4.29e-2, 3.52e-3, 3.4e-4, 7e-6 against 4.24e-2, 3.42e-3, 3.2e-4,
   8e-6 without it.
3. **RNN-FFWD: the "recurrent" network is not recurrent.** RNN.C copies the delayed outputs into
   inputs nodes_0+1…, then calls `ffwd_sig(..., nodes_0+1, ...)`, which stops before them. The
   feedback never reaches the nodes, so the network that was simulated is a single tanh neuron
   (plus RTRL bookkeeping for weights that have no effect). This **biased the comparison against
   the RNN**. With the fix, the telephone-channel RNN improves at every SNR above 6 dB (e.g. 16 dB:
   6.6e-3 → 4.0e-3; 20 dB: 3.2e-4 → 8.5e-5), and on channel 1 it beats both the buggy RNN and a
   2-tap ADALINE (10 dB: 2.5e-3 vs 6.2e-3 and 6.1e-3). It still loses to the ADALINE on the
   telephone channel above 14 dB (20 dB: 8.5e-5 vs 9e-6). The report's RNN numbers match the
   *buggy* program closely, so the buggy version is the one behind Fig 9.3.
4. **RNN-KRON: RTRL term counted nodes_1 times** (the δ_kj·u_l term sits inside the sum over
   fed-back nodes). The effect is to scale the learning rate by nodes_1. The learning rates in the
   report were tuned with the bug present, so fixing it alone makes the RNN under-train (telephone
   channel 20 dB: 3.2e-4 → 5.2e-2). It is a real bug but is better treated as part of the tuning.
   The all-fixes `clang_fix` RNN is therefore worse than the RNN-FFWD-only variant.
5. **Learning rate 0.08 for the telephone ADALINE diverges.** The step limit for 15 taps with unit
   input power is about 1/15 ≈ 0.067. Every trial blows up. 0.008 reproduces Fig 9.3. This is a
   reporting error, not a program error.
6. **Endless retraining.** LMS.C/RNN.C retrain `while(fail)`, where fail means the mean squared
   error of the last 400 bits ≥ 0.6, and there is no limit on retries. Settings that cannot pass
   never finish: Fig 5.4 η=0.0001 (still about 0.68 at bit 1000) and the whole telephone ADALINE
   at η=0.08. For η=0.1 the program needs 11.7 retrains per accepted trial, so the Fig 5.4 "0.1"
   curve is an average over the lucky 8% of runs. Neither curve in Fig 5.4 can have come from this
   program unmodified. The retrain rule also means published error curves only include runs that
   passed, which flatters every network the same way.
7. **Clock seeding.** `srand(time())` runs before every trial and every retry. Trials that start in
   the same second reuse the same random numbers (only the preamble phase differs, because the PN
   register is never reset). A failing retry within the same second repeats the same failure. On
   the hardware of the time a trial took about 1–4 s (DOSBox estimate), so most 1995 trials probably
   did get distinct seeds. How many collided is unknown. On a modern PC the trials would collide,
   which is why every rebuild here uses `EQ_SEED`.
8. **31-bit training sequence.** The "1000-bit preamble" is the 5-bit PN generator's 31-bit pattern
   repeated about 32 times. A linear LMS does not mind. The RBF only ever sees 31 of the 2048 symbol
   patterns that the telephone channel needs, so section 9's RBF failure was partly decided by the
   preamble before network size came into it.
9. **BPBC debug leftovers (BP-DEBUG).** Two weight dumps with "press a key" waits are outside
   `#ifdef DEBUG`, and they call `print_weights` with its layer arguments swapped, which dereferences
   the input layer's NULL weight pointers. On real-mode DOS that reads low memory and does no harm;
   on a hosted OS it segfaults. The printed BPBC is therefore a debug build. `e:\temp\` paths are
   hard-coded (on macOS they become literal file names in the current folder).
10. **`bp_tidyup` use-after-free**: it frees the layer arrays, then reads `layer[i].weights` to free
    those. This happens at exit, so it has no effect on results.
11. **`for(i=i; ...)` in RB_INIT.H** is harmless: `i` already equals nodes_1/2, so it fills the
    second half of the output weights with +1 as intended.
12. Minor: `void main` returns a garbage exit status. RNN.C strips `strlen("LMS.EXE")` from argv[0]
    (it works only because "RNN.EXE" has the same length). `lms_tidyup` never frees `error_run`.

### Sanitizer sweep (`src/clang/sanitize.sh`, output in `results/sanitize.txt`)

ASan+UBSan builds with the heap guard off, run on short versions of every program's scenarios:

| program | faithful build | fixed build |
|---|---|---|
| LMS | heap-buffer-overflow in `fill_rand` (bug 1), both channels | clean |
| RNN | clean | clean |
| RB | clean | clean |
| BPBC | NULL member access and SEGV in `print_weights` (bug 9) | heap-use-after-free in `bp_tidyup` (bug 10), at exit |

macOS has no MemorySanitizer, so uninitialised reads were checked by reading the code instead. All
arrays come from `calloc`, and the only "uninitialised" value that matters is bug 2.

**Memory-safe version: the Python line port**, chosen over a C++ `std::vector::at()` rewrite.
It already exists, it matches the C bit for bit, and every array access is bounds-checked by the
language (an out-of-range index raises `IndexError`). A C++ port would mean a second translation
with its own risk of new mistakes. The one real overrun (bug 1) is handled explicitly: by default
the write is dropped (the harmless outcome), and `EQ_STRICT=1` raises an error naming it. Because
Python and C agree everywhere, the bounds-checked runs show that apart from the corrupted first
error point, **no 1995 result depended on a memory fault**. The fixes that change results are the
logic fixes (bugs 3 and 4) and the reporting fixes (5 and 6), not the memory ones.

## Verdict (plain English)

The report's headline measurements stand up. Rebuilt from its own listings and run on four
different systems, which all agree with each other, the programs reproduce its main results: the
standard equaliser's error rates on the simple channel, and both curves of the final telephone-line
comparison. That holds once one misprint is corrected (a setting of 0.08 that must have been 0.008,
because 0.08 makes the program run forever). The comparison was not a fair test of the neural
networks, though. The "recurrent" network had a programming mistake that switched off the feedback
which makes it recurrent. The simple-channel tests gave the standard equaliser seven inputs against
the networks' two, and at equal size the networks were as good or slightly better. The radial-basis
network was trained on a 31-step pattern that could never show it most of the telephone line's
behaviour. Even with the feedback bug fixed, the standard equaliser still wins on the realistic
telephone channel, so the practical conclusion for that job, "use the simple method", is solid. The
report's conclusions were explicitly limited to practically sized networks at 1995 computing cost, and on that basis they stand; the bugs, if anything, undersold the networks. Only the one-line summary reads more broadly than that. It is also
uncertain how some figures were made: the simple-channel neural-network figures show about 3–10
times more errors than their listed programs produce, and two learning-rate curves cannot have come
from the listed code unmodified.

## DOS runs (16-bit Open Watcom builds under DOSBox-X)

`src/openwatcom/build.sh` rebuilt LMS/RNN/RB/BPBC.EXE (large model, x87, with the kbhit/getch shim so
BPBC's debug pauses do not wait for a key). `scenarios/run_dos.py` ran every scenario in headless
DOSBox-X (cycles=max, dynamic core, EQ_SEED=1000) on the Mac mini, 6 instances at a time:
100 trials for LMS/RNN/RB, 20 for BPBC (as native). 82 jobs, 64 min wall clock, but 5 of those
jobs were the printed η=0.08 telephone ADALINE, which never finishes (bug 5) and hit the 1 h timeout;
it is now skipped. Without it the run is a few minutes. Per job, at max cycles on an M4: LMS 8-14 s,
RNN toy 22 s, RNN telephone 75 s, RB 20 s, BPBC 20 trials 52 s.
Full table: `dos/compare.md` (`compare_dos.py`); data `dos/dos.csv`, curves `dos/curves/`.

- **LMS, RNN, RB: 72 of 72 completed jobs match the native clang build.** BER and fail counts are
  identical, and the learning curves agree to within 1e-6 (print precision) from line 2 on. Line 1 of
  ERRORS.TXT differs by design: the fill_rand overrun clobbers `errors[0]` on the DOS heap. RB writes no
  curve, so it is compared on BER only. The 1995-format tables above therefore hold for DOS too.
- **BPBC, faithful listing:** runs to completion in DOS. On macOS the same code crashes in the debug
  `print_weights` call (swapped args, NULL weights; see BP-DEBUG). Under real-mode DOS the bad pointer
  reads garbage and prints it, but does not crash. That is presumably why the bug went unnoticed in 1995.
  Channel 1 at 10/14/21 dB is identical to the BP-DEBUG-fixed native build (1.57e-2, 7.95e-4, 1e-5).
  At 6 dB it is 6.80e-2 against 6.79e-2. Channel 2, which does not train at all (no decision delay),
  gives 0.34 / 0.20 / 0.12 / 0.13 against native 0.34 / 0.21 / 0.11 / 0.13. These small differences are
  floating point, not logic: 80-bit x87 intermediates against arm64 doubles, amplified by 1000 epochs of
  batch training on the runs that do not settle. The conclusions about backprop are unchanged.
- Endless retrain (`while(fail)`, no cap) behaves the same in DOS: fig5_4 η=0.0001 and fig9_3 η=0.08 never finish.
