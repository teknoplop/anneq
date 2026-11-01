# The 1995 scenarios rerun with the bugs fixed

Issues teknoplop/anneq#10 to #18 (#19 is not a code bug; see the end). "Unfixed" numbers are
the 1995 programs as listed (`results/rebuild/`, clang build = Python line port = DOS, 100 trials).
"Fixed" numbers are from the Python line port with every fix on (`src/anneq_fixed.py`, its default;
was `EQ_FIXED=1`), in `results/fixed/` (same CSV format, plus `gaveup_per_trial`, `final_err_all`, `secs`). They are
100 trials (BPBC 20) with EQ_SEED=1000. The full side-by-side table for every point is
`fixed/compare.md` (`scenarios/compare_fixed.py`), and the plots are `fixed/*.png`
(`scenarios/plot.py --out fixed`). Report section and figure numbers are those in
`docs/report_text.md`. "Report" values are read off the 1995 figures, so they are approximate.

## What the fixed variant changes

All fixes are in `src/anneq_fixed.py`, on by default and selectable by tag (`EQ_FIX=TAG,..` for only
those, `EQ_NOFIX=TAG,..` to drop some). The 1995 code paths are `src/anneq_orig.py`.

| tag | issue | change | C counterpart |
|---|---|---|---|
| RNN-FFWD | #10 | feed all n0+n1+1 inputs (incl. the fed-back outputs) to the RNN nodes, in training and test | yes |
| RNN-KRON | #11 | δ·u term of RTRL counted once (as in C: divided by nodes_1 inside the sum); η re-tuned, see below | yes |
| LMS-FILL | #12 | `fill_rand` fills nodes_0+1 weights, not nodes_0+2 | yes |
| LMS-BIAS | #13 | ADALINE bias input set to −1 (Fig 5.1) | yes |
| BP-DEBUG | #17 | BPBC debug weight dumps / key waits removed (screen only; never ported) | yes |
| RETRY-CAP | #14 | at most 10 training attempts per trial; GAVEUP = share of trials that never passed. Their BER still counts. The curve averages the trials that passed (the 1995 convention), and `final_err_all` / `curves/*_all__*` average every trial's last attempt | no |
| SEED-ONCE | #15 | one `srand` at start (EQ_SEED), never the clock per trial or retry | no |
| RAND-PREAMBLE | #16 | random ±1 training bits instead of the repeated 31-bit PN pattern | no |
| BP-PATHS | #17 | BPBC files are BP_CONST.TXT etc. in the run folder, not `e:\temp\` | no |
| MINOR | #18 | explicit exit status 0. The rest of #18 (argv[0] path stripping, `for(i=i;…)`, missing `int bit_no`, `error_run` never freed, `bp_tidyup` use-after-free) has no counterpart in the Python port | no |

`pyfix_pn31` in the CSVs is "every fix except RAND-PREAMBLE". It separates the effect of the
training data from the code fixes.

**RNN learning rate (#11).** #11 turns out to be a pure learning-rate scale: with #10 fixed, results with and without #11 are identical once η is multiplied by nodes_1 ([RNN_ISOLATION.md](RNN_ISOLATION.md)). #10 alone, with η re-tuned, changes the conclusion. Once the δ·u term is counted once, the 1995 rates (0.005 with 3 nodes,
0.003 with 4) are effectively nodes_1 times smaller and the RNN under-trains. I swept η with every
fix on, using 20 trials at two SNRs per channel (`fixed/rnn_eta_sweep_{toy,tel}.csv`). The table
shows the mean BER over those two SNRs:

| η | 0.003 | 0.012 | 0.03 | 0.05 | **0.08** | **0.12** | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| channel 1, 2-in/3-node (10, 14 dB) | 2.5e-3 | 1.5e-3 | 1.05e-3 | 9.0e-4 | 8.4e-4 | **8.0e-4** | 8.4e-4 | 9.7e-4 | 1.2e-3 |
| telephone, 10-in/4-node (14, 18 dB) | 7.1e-2 | 7.6e-3 | 4.7e-3 | 4.2e-3 | **4.2e-3** | 5.0e-3 | 9.0e-3 | 1.7e-2 | 3.9e-2 |

Chosen: **η = 0.12** (channel 1) and **η = 0.08** (telephone). There is a broad optimum. The first
retrains appear on the telephone channel at η = 0.3 (0.05 per trial) and 0.5 (0.2 per trial); on
channel 1 there are none up to 0.5. The sweep uses the same seed stream as the 100-trial runs, so
the chosen η is very slightly flattered. To check the comparison was fair, I swept the ADALINE as
well (`fixed/ada_eta_sweep_tel.csv`, 15 taps). Its 0.008 (the report's "0.08" corrected) is
at its optimum: 14 dB 1.19e-2 (best of the sweep), and 18 dB 2.95e-4 against a best of 2.6e-4 at
0.012. Retrains start at 0.024, and at 0.032 one trial in five never trains at 14 dB.

## Bit-exact checks

`src/tools/check_pyport.py` (log `results/check_pyport.txt`) runs 28 cases (all four programs, retry
and delay cases, the telephone channel). It compares every output file byte for byte, as follows:

- **Unfixed Python vs the 1995 C** (`src/clang/build`; BPBC vs `build_fix/bpbc`, because the faithful
  C BPBC segfaults on a hosted OS and BP-DEBUG only removes screen output): all identical.
- **Each C-counterpart fix vs its one-fix C build** (`build_fix_LMS-FILL`, `_LMS-BIAS`,
  `_RNN-FFWD`, `_RNN-KRON`): all identical.
- **All C fixes together vs `src/clang/build_fix`** (`make fixed`) for LMS, RNN and BPBC: all identical.

Two subtleties came up, and the checks now cover both:

1. `URAND` is the unbracketed macro `(double)rand()/RAND_MAX`. So in C `2.0*M_PI*URAND` (noise)
   is `(2π·rand)/RAND_MAX`, and `0.3*URAND` (BPBC weights) is `(0.3·rand)/RAND_MAX`. Those round
   differently from `2π·(rand/RAND_MAX)`. The Python port had the bracketed form in `noise()`, so
   noise samples were sometimes 1 ulp off. The LMS/RNN/RB outputs print 7 digits and never showed
   it, but BPBC prints 15 digits after 1000 epochs and did. Both are now ported as C evaluates them.
2. The src/clang default build lets clang fuse a·b+c into one FMA instruction on arm64. C allows that.
   The 1995 x87 code and Python round every step separately, so they cannot fuse. The checker
   therefore compares against references compiled from the same prepared sources with
   `-ffp-contract=off`. It also prints the comparison with the default binaries: identical for LMS,
   RNN and RB, and different in BPBC's last digits. The BPBC numbers in `results/rebuild/` are from
   the FMA build, which accounts for some of the "x87 vs arm64" BPBC drift in RESULTS.md.

The numpy port is not used for these runs. It draws from numpy's own generator, so it is not
bit-exact with C. The line port was fast enough anyway.

## Run times (Mac mini, Apple M4, 8 workers, CPython 3.12 via uv, `~/anneq-runs/pyfix/`)

`REMOTE=mini.local uv run --no-project python scenarios/run_fixed.py --jobs 8`. That is 168 jobs
(84 scenario points × {pyfix, pyfix_pn31}), 100 trials each (BPBC 20), in **115 s** wall time and
about 900 CPU-s. The longest job was the telephone RNN at 23 s per 100 trials; 20 BPBC trials took
up to 10 s. The η sweeps (60 jobs, 20 trials) took 18 s. In 1995 the same telephone RNN point would
have taken hours.

## Figure by figure

### Fig 5.3: ADALINE, number of inputs (§5.3), channel 1, 21 dB, η=0.01

Final error, 1/2/3/7/11 taps: unfixed 0.212 / 0.058 / 0.022 / 0.0117 / 0.0126; fixed 0.213 / 0.058 /
0.024 / 0.0120 / 0.0129 (report ≈ 0.2 / 0.07 / 0.025 / 0.015 / 0.015). The first ERRORS.TXT point is
now right in DOS too (LMS-FILL). **Unchanged:** "increasing the number of inputs reduces the final
error … 7 inputs" holds.

### Fig 5.4: ADALINE, learning rate (§5.3)

| η | unfixed | fixed |
|---|---|---|
| 0.1 | 11.7 retries per accepted trial; curve = the lucky 8 % that passed, final error 0.46 | **0 of 100 trials train in 10 attempts**; mean over all trials diverges (final error 3.5e56); BER 0.47 |
| 0.01 | final 0.0117 | final 0.0120 |
| 0.001 | final 0.077 | final 0.082 |
| 0.0001 | never finishes (no cap) | no trial reaches MSE < 0.6 in 1000 bits; all-trial final error 0.66 (report ≈ 0.8) |

**Unchanged, and better supported:** "0.1 makes the network unstable, and the error never
converges", "0.0001 … converge is very slow", "optimum 0.01". With the bias input switched on, η=0.1
is past the LMS stability limit (8 inputs). The fixed program shows both curves of Fig 5.4 as the
report describes them, which the 1995 listing cannot produce (#19).

### Fig 5.5 and Fig 5.7: ADALINE BER, channels 1 and 2 (§5.3, §5.4)

| SNR dB | report | unfixed ch1 | fixed ch1 | fixed ch1, 31-bit PN | unfixed ch2 (delay 7) | fixed ch2 |
|---|---|---|---|---|---|---|
| 6 | ~4e-2 | 4.24e-2 | 4.41e-2 | 4.25e-2 | 4.19e-2 | 4.43e-2 |
| 10 | ~3.5e-3 | 3.42e-3 | 3.60e-3 | 3.42e-3 | 3.19e-3 | 3.52e-3 |
| 12 | ~3e-4 | 3.2e-4 | 3.8e-4 | 3.5e-4 | 3.0e-4 | 3.6e-4 |
| 14 | ~7e-6 | 8e-6 | 1.4e-5 | 7e-6 | 8e-6 | 1.0e-5 |

**Unchanged:** "very acceptable performance in noisy conditions" and "equally as effective on this
channel [2]". The fixed values are a few percent higher. Most of that comes from the random training
data (with the PN they match the unfixed values), because a balanced PN is a slightly better LMS
training signal than coin flips. The bias input on its own changes nothing measurable (#13).

### Figs 6.2–6.4: batch backprop, BPBC (§6.3), 2-6-6-1, 40 samples, 1000 epochs, lr 0.1 / mom 0.5 assumed

| SNR dB | unfixed ch1 | fixed ch1 | fixed ch1, 31-bit PN | unfixed ch2 | fixed ch2 |
|---|---|---|---|---|---|
| 6 | 6.8e-2 | 5.9e-2 | 5.8e-2 | 0.34 | 0.35 |
| 10 | 1.6e-2 | 1.25e-2 | 1.19e-2 | 0.21 | 0.21 |
| 14 | 8.0e-4 | 4.1e-3 | 1.6e-3 | 0.11 | 0.16 |
| 21 | 1e-5 | 6.0e-3 | 2e-5 | 0.13 | 0.20 |

**Unchanged:** "converged well in noisy conditions … for channel 1" (Fig 6.2, training error
≈1e-6). Channel 2 still does not train. BPBC has no decision delay, which is a design limit, not one
of the listed bugs, so the report's channel-2 curve in Fig 6.2 and the boundary in Fig 6.3(b) are
still not reproduced. With only 40 training samples, random bits are worse than the PN: some draws
under-represent a pattern, and one trial in 20 then fails at 21 dB (hence 6e-3). For this experiment
the 31-bit PN of §3.3 is the better choice.

### Fig 7.2 and Fig 7.3: RNN, channel 1 (§7.3), 2 inputs, 3 nodes

Fig 7.2 (21 dB) final ensemble error: report ≈ 0.055, unfixed 0.021, **fixed 0.0016**.

| SNR dB | report | unfixed RNN | fixed RNN (η 0.12) | ADALINE 2 taps (fixed) | RBF ch1 (fixed) |
|---|---|---|---|---|---|
| 6 | ~1.1e-1 | 4.26e-2 | 3.33e-2 | 4.38e-2 | 3.65e-2 |
| 8 | ~7.5e-2 | 1.87e-2 | 9.4e-3 | 1.90e-2 | 1.39e-2 |
| 10 | ~4.5e-2 | 6.24e-3 | **1.50e-3** | 6.29e-3 | 3.71e-3 |
| 12 | ~2.2e-2 | 1.41e-3 | 9.3e-5 | 1.46e-3 | 6.2e-4 |
| 14 | ~8e-3 | 1.78e-4 | 0 | 1.46e-4 | 4.2e-5 |

With the feedback connected and the learning rate re-tuned, the RNN converges about 13 times lower
(Fig 7.2) and makes 4 times fewer errors at 10 dB. It now beats the RBF and the equal-size ADALINE at
every SNR. **Unchanged:** "converge well in noisy situations". **Changes:** "the RNN was very
sensitive to the values for the learning rate … If the learning rate was too high then the RNN
became unstable very quickly" (§7.3). The fixed RNN has a broad optimum: BER within about 20 % of the
best over η = 0.05–0.3 on channel 1 and 0.03–0.12 on the telephone channel. Failed trainings appear
only on the telephone channel, from η = 0.3, and never on channel 1 up to 0.5. A rewrite should say it is moderately sensitive, not very
sensitive. (The structure-size claim was not retested.) The learning rate quoted in §7.3 (0.005)
belongs to the bugged RTRL; for the corrected algorithm it is 0.12.

### Fig 8.3: RBF, channels 1 and 2 (§8.3)

Channel 1 at 6/10/14 dB: unfixed 3.66e-2 / 3.84e-3 / 5.8e-5, fixed 3.65e-2 / 3.71e-3 / 4.2e-5.
Channel 2: 0.213 / 6.67e-2 / 6.41e-3 against 0.213 / 6.68e-2 / 6.45e-3. **Unchanged.** RB.C has
none of the code bugs, and neither the seeding nor the training data matters to it on a 2-tap
channel: the 31-bit PN already covers all 8 states. "Performs extremely well" and "the training
preamble can be short" stand. One thing changes in comparison with §7: the fixed RNN is now better
than the RBF on channel 1.

### §9, Figs 9.2 and 9.3: telephone channel, ADALINE 15 taps vs RNN 10 inputs / 4 nodes

| SNR dB | report ADALINE | report RNN | unfixed ADALINE (η .008) | unfixed RNN | **fixed ADALINE (η .008)** | **fixed RNN (η .08)** | fixed RNN, 31-bit PN | fixed ADALINE 10 taps |
|---|---|---|---|---|---|---|---|---|
| 6 | ~1.4e-1 | ~1.4e-1 | 1.45e-1 | 1.38e-1 | 1.49e-1 | 1.49e-1 | 1.48e-1 | 1.43e-1 |
| 10 | ~6e-2 | ~6e-2 | 6.18e-2 | 6.41e-2 | 6.39e-2 | 6.12e-2 | 6.14e-2 | 6.11e-2 |
| 14 | ~1.1e-2 | ~1.7e-2 | 1.10e-2 | 1.73e-2 | 1.16e-2 | **8.5e-3** | 9.8e-3 | 1.09e-2 |
| 16 | ~2.3e-3 | ~6.5e-3 | 2.30e-3 | 6.57e-3 | 2.67e-3 | **1.48e-3** | 1.86e-3 | 2.45e-3 |
| 18 | ~2.4e-4 | ~2e-3 | 2.55e-4 | 1.78e-3 | 3.0e-4 | **1.16e-4** | 1.45e-4 | 2.8e-4 |
| 20 | ~6e-6 | ~3.5e-4 | 9e-6 | 3.23e-4 | 7e-6 | **1e-6** | 3e-6 | 1.1e-5 |

Final ensemble error (Fig 9.2), ADALINE / RNN: unfixed 0.56 / 0.43 at 6 dB, 0.19 / 0.16 at 14 dB,
0.070 / 0.11 at 20 dB. Fixed 0.56 / 0.49, **0.20 / 0.050**, **0.075 / 0.0078**.

- **The printed ADALINE rate 0.08 still diverges.** With the retry cap the program now ends and
  reports 100 % of trials failing (BER ≈ 0.5 at every SNR). 0.008 is needed (a misprint, RESULTS.md
  bug 5). **Unchanged:** a rewrite should give 0.008.
- **Fig 9.3, "the ADALINE network outperformed the RNN equaliser for BERs": reversed.** With the
  corrected RNN the two are level up to 10 dB. From 14 dB up the RNN makes fewer errors: 1.4× at
  14 dB, 1.8× at 16 dB, 2.6× at 18 dB, about 7× at 20 dB (1 against 7 errors in 10^6 bits, so that
  last ratio is rough). That holds against the ADALINE at its own best rate, and against a 10-tap
  ADALINE with the same inputs as the RNN. It holds with the 1995 PN preamble too (pyfix_pn31).
- **Fig 9.2, "convergence … very similar with the RNN achieving a slightly smaller final error":
  true only at low SNR.** At 6 dB (which matches the report's curve levels of ≈0.6 / 0.5) it still
  holds: 0.56 against 0.49. From 10 dB up the corrected RNN ends 2–10 times lower. The SNR used for
  Fig 9.2 is not stated; a rewrite should state it, and if it was above 6 dB, "slightly smaller"
  becomes "much smaller".
- §9's RNN parameters change: "10 inputs, 4 output nodes and a learning rate of 0.003" becomes
  0.08 with the corrected RTRL.
- **Unchanged:** "No optimum parameters could be found for the backpropagation equaliser" and
  "the RBF equaliser would also not train on the real life channel model … 2048 co-ordinate points".
  Neither was among the scenarios, and the fixes do not touch the network-size argument. The random
  preamble removes the restriction to 31 of the 2048 patterns (#16), but an RBF still needs up to
  2^11 centres. That argument stands. The preamble was an extra handicap, not the deciding one.

## Which statements of the report change

Would change with the corrected code:

1. **Summary (p. 2):** "it is concluded that artificial neural networks are unsuitable of channel
   equalisation". The corrected RNN beats the ADALINE on the realistic channel above 10 dB (Fig 9.3)
   and on channel 1 at equal size (Fig 7.3), so the blanket statement no longer follows. It would
   need narrowing to the RBF and backprop networks at practical sizes.
2. **§7.3:** "the RNN was very sensitive to the values for the learning rate … became unstable very
   quickly". It is moderately sensitive, with a broad optimum and failed trainings only at η ≥ 0.3 (telephone channel). The
   quoted η of 0.005 becomes 0.12 (corrected RTRL).
3. **§7.3, Figs 7.2 and 7.3 (curves and values):** the corrected RNN converges to ≈0.0016 (not
   ≈0.055) and its BER is 1.5e-3 at 10 dB. Both figures would need redrawing. They already
   disagreed with the listing (#19), and the fix widens the gap.
4. **§9:** "The RNN equaliser had … a learning rate of 0.003". This becomes 0.08.
5. **§9, Fig 9.2:** "very similar with the RNN achieving a slighter smaller final error". This holds
   at 6 dB only; at 10–20 dB the RNN's final error is 2–10 times smaller.
6. **§9, Fig 9.3:** "the ADALINE network outperformed the RNN equaliser for BERs". This is reversed
   above 10 dB (RNN 1.5e-3 vs 2.7e-3 at 16 dB, 1.2e-4 vs 3.0e-4 at 18 dB) and level below.
7. **§10 Conclusions:** "its high sensitivity to starting parameters and instability made it second
   best to the ADALINE network" and "the three ANNs compared to the ADALINE network showed no
   advantage in using complex non-linear structures". Both fail for the RNN: it is the best of the
   equalisers tested on the telephone channel above 10 dB. Its remaining cost is computation (RTRL needs
   about 30 times the ADALINE's multiply-adds per training bit; here 0.22 s against 0.03 s per trial), which §10's
   "practically sized … computation speed" framing can still weigh against it.
8. **§3.2 (procedure):** "If this test is failed, then the ANN is reset and trained again". This
   should add that retraining is capped (10 attempts) and the failure rate reported. In the corrected
   runs no trial gives up at the report's settings. Retries happen only at 6 dB (0.18 per trial for
   the ADALINE, 0.06 for the RNN on the telephone channel).

Would not change:

- §5.3 / Fig 5.3: 7 inputs is enough. §5.3 / Fig 5.4: 0.1 unstable, 0.0001 very slow, 0.01
  optimum. With the retry cap these now come out of the program itself.
- §5.3 / Fig 5.5 and §5.4 / Fig 5.7: ADALINE BERs and "equally as effective on channel 2" (within a
  few percent; identical with the PN preamble).
- §5.2: "excluding the bias node, the ADALINE equaliser is identical to the FIR … filter". The bias
  now works and makes no measurable difference.
- §6.3 / Fig 6.2: batch backprop converges on channel 1 in noise. Channel 2 still does not train,
  which is not a bug fix matter (no decision delay).
- §8.3 / Fig 8.3: RBF results and "performs extremely well … preamble can be short".
- §3.3: a 31-bit PN gives enough training patterns for the 2-tap channels. It does for every
  network tested; random bits are no better and worse for the 40-sample BPBC.
- §9: the ADALINE's Fig 9.3 curve (with 0.008 in place of the printed 0.08), the backprop and RBF
  statements, and the 2048-point argument.
- §10: "the RBF and backpropagation ANNs are not suitable for equalisation" at practical size.

## #19 (toy-channel figures 7.3 / 8.3 show 3–10× more errors than the listing)

The fixes do not resolve it. For the RBF (Fig 8.3) nothing changes, and the listing still makes 3–10
times fewer errors than the figure. For the RNN (Fig 7.3) the gap gets wider: at 10 dB the report
reads ≈4.5e-2, the listing 6.2e-3, the fixed program 1.5e-3. So those figures were not made by a
bug-fixed version of this code either. The Fig 5.4 part of #19 is resolved: a capped retry loop
(or no retry), with curves averaged over all trials, gives the "never converges" η=0.1 curve and the
"very slow" η=0.0001 curve the report describes.
