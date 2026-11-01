# Neural Network Based Adaptive Equalisation Techniques — a rewrite

*A shorter retelling of a 1995 final-year BEng project thesis (Electronics Engineering with Communications, dated 28 April 1995). Numbers are taken from the original report; values read off its graphs are approximate and marked "about".*

---

## 1. TL;DR for an accountant

When data is sent down a telephone line, each pulse smears into the next one, so the receiver can misread what was sent. Receivers fix this with a small correcting circuit, the "equaliser", which has to learn how the line behaves by listening to a known test message first. In the early 1990s there was a lot of excitement about "neural networks", computer programs loosely modelled on the brain, and several research papers suggested they might do this correcting job better than the standard method. This project wrote simulations on a PC of three types of neural network, set each one against the standard, long-established method on a realistic model of a British telephone line, and counted how many bits each got wrong. The standard method won: two of the neural networks could not be made to work at a practical size, and the third worked but made more errors and was harder to set up. The report concluded that, with the methods and computers of the day, neural networks were not worth using for this job. That mattered because modem makers were weighing extra cost and complexity against claimed gains, and this was evidence that the simple approach was still the right one.

**Why revisit it?** The revisit had two purposes: to exercise today's AI tools on a real project, 30 years on (both the Python machine-learning libraries and AI-assisted coding), and to find the bug the author had always suspected was there.

**Was the project right? (2026 re-check.)** The programs were rebuilt from the report's own listings and run on four different systems, and those runs agree with each other and reproduce the report's main measurements, once one misprinted setting is corrected (0.08 should be 0.008). So the numbers were genuine. But the head-to-head was tilted against the neural networks: a programming mistake switched off the feedback that makes the "recurrent" network recurrent, the simple-channel tests gave the standard method seven inputs against the networks' two (at equal size the networks were as good or slightly better), and one network was trained on a pattern too short to show it most of the telephone line's behaviour. With all the bugs fixed and its learning rate re-tuned, the recurrent network makes fewer errors than the standard method on the realistic telephone line at the better signal levels, but it needs about 30 times the computation to train, so the practical advice for a modem with a fixed budget and a short test signal — use the simple method — holds. The report's actual conclusion was narrower than its one-line summary: networks small enough to compete with the standard method on computation (the binding constraint for a 1995 modem) showed no advantage, and it said plainly that the work only scratched the surface and that better structures or algorithms might change that. Read that way, the conclusion stands; the bugs mean the networks were, if anything, slightly undersold. Some of the neural-network graphs show more errors than the listed programs produce, so it is unclear exactly how they were made.

---

## 2. The rewrite

### 2.1 The problem: intersymbol interference

A digital link sends a stream of symbols, here simply +1 or −1, one every symbol period *T*. A real channel does not deliver each symbol cleanly inside its own time slot. A telephone line disperses it in time; an HF radio link delivers several delayed copies (multipath). Either way, by the time a symbol arrives it has spread into the slots of its neighbours. What the receiver samples at any moment is therefore a mix of the current symbol and bits of the previous few. That overlap is **intersymbol interference (ISI)**, and it gets worse the faster you signal. Add ordinary random noise on top and the receiver starts making wrong decisions.

### 2.2 The fix: adaptive equalisation

An **equaliser** is a filter at the front of the receiver that tries to undo the channel's smearing and recover the original stream. The difficulty is that the channel is not known in advance and can drift, so the equaliser must **adapt**: it is trained on a known preamble at the start of a call, compares its output with what it knows was sent, and adjusts itself to shrink the error.

The established design since the mid-1960s was the **transversal (FIR) equaliser**: keep the last few received samples in a delay line, multiply each by a weight, and add them up. The weights are tuned by the **LMS** (least mean squares) algorithm. Its limitation is that it is linear — it can only draw a straight-line boundary between "this was a +1" and "this was a −1". Non-linear designs such as the Decision Feedback Equaliser followed in the late 1960s, and when neural networks came back into fashion in the mid-1980s researchers began applying them too. This project set out to compare several neural networks against the linear baseline under realistic conditions.

### 2.3 How the channel was simulated

The channel is modelled as:

> received signal = (transmitted signal passed through the channel filter) + noise

- **The channel filter** is itself an FIR filter: the received sample is a weighted sum of the current and previous transmitted symbols. The weights describe how much of each earlier symbol leaks into the present one. Two toy channels were used during development:
  - Channel 1: `H[z] = 1.0 + 0.5z⁻¹` (current symbol plus half the previous one)
  - Channel 2: `H[z] = 0.5 + 1.0z⁻¹` (the previous symbol dominates — harder, see below)
  - The "real life" channel, a typical British telephone line taken from a textbook, has ten coefficients: 0.047, 0.321, 0.783, 0.427, 0.293, 0.100, 0.041, 0.038, 0.020, 0.011. Note the peak is two symbols in, with a long tail.
- **Noise** is Gaussian with zero mean. Signal-to-noise ratio is set in decibels, and the noise variance follows as σ² = 10^(−SNR/10) (signal power is 1). Because Borland C had no Gaussian random generator, a standard formula built Gaussian samples from pairs of uniform `rand()` values; a histogram of 8,192 samples was checked against the ideal bell curve.
- Simple baseband ±1 pulses were used rather than real modem modulation (FSK, QPSK) because ISI is easiest to see that way.

### 2.4 The test method

Every equaliser went through the same two-stage test:

1. **Training.** 1,000 bits of a known preamble are sent and the squared error recorded bit by bit. The preamble comes from a 5-bit pseudo-random generator — a shift register with bits 2 and 5 XORed back into the input — which produces a fixed 31-bit pattern with roughly equal +1s and −1s. An identical generator at the receiver produces the "desired" signal to compare against.
2. **Bit-error test.** 10,000 random bits are sent; the equaliser output is forced to +1 or −1 by a hard limiter (sign of the output), and wrong decisions are counted. **BER** (bit error rate) = errors ÷ 10,000.

The pair was repeated 100 times. Averaging the 100 squared-error traces gives the **ensemble squared error** curve, which shows how quickly and how far each equaliser learns (its convergence). The BER was averaged too. A network that obviously failed to train was reset and retrained, so that one-off failures did not distort the averages.

### 2.5 Seeing equalisation as classification

The useful insight in the report is to treat equalisation as a sorting problem. Take a two-input equaliser on channel 1 with no noise. The pair of samples it sees, (x(n), x(n−1)), depends on the last three transmitted symbols, so there are only 8 possible points, e.g. (−1.5, −1.5), (−0.5, +0.5), (+1.5, +0.5). Four of them mean "+1 was sent now", four mean "−1". Noise blurs each point into a small cloud.

The equaliser's job is to draw a boundary between the two groups of clouds. The best possible boundary (the **Bayesian** optimum, the one that minimises errors given the noise) can be calculated and plotted, and this is the target any network is trying to approximate.

- On **channel 1** the optimum boundary can be straightened into a single line: the groups are *linearly separable*, so a linear equaliser can in principle do the job.
- On **channel 2** they are not: the +1 and −1 points are interleaved and the optimum boundary zig-zags. That is the case where a non-linear network should, in principle, beat a linear one.

### 2.6 The baseline: ADALINE trained by LMS

The **ADALINE** (Adaptive Linear Element, Widrow, early 1960s) is a single "neuron" with no squashing function: output = weighted sum of the inputs. Fed from a delay line, it is exactly the classic transversal equaliser with a bias term, which is why it served as the reference.

LMS updates each weight after every bit:

> new weight = old weight + η × error × the input on that weight

In words: if the output was too low and this input was positive, nudge its weight up, in proportion to both. η, the learning rate, sets the step size. Appendix A of the original derives this: the average squared error, plotted against the weights, is a smooth bowl with one bottom; LMS walks downhill using each single sample as a noisy estimate of the slope.

**Results (channel 1, 21 dB SNR):**
- Inputs: 1, 2, 3, 7 and 11 taps were tried. More taps gave lower final error — settling at about 0.2 (1 tap), 0.07 (2), 0.025 (3) and about 0.015 for 7 and 11, which were indistinguishable. **7 taps** was chosen; beyond that only complexity rises.
- Learning rate (7 taps): **0.1 was unstable** and never converged; **0.0001** was stable but barely moved in 1,000 bits (would need tens of thousands); **0.001** was still falling slowly at bit 1,000; **0.01 was best**, reaching about 0.015 within about 250 bits.
- BER (7 taps, η = 0.01): about 4×10⁻² at 6 dB, 3.5×10⁻³ at 10 dB, 3×10⁻⁴ at 12 dB and about 7×10⁻⁶ at 14 dB.

**Channel 2** looks like a problem for a linear device, but the report shows a standard way round it. An ideal equaliser would be the channel's mathematical inverse. For channel 2 that inverse needs *future* symbols, which is impossible — unless you accept a delay. Train the equaliser to reproduce the symbol sent a few periods ago rather than the current one, and a truncated, delayed inverse becomes realisable by an FIR filter. With 7 taps, η = 0.01 and the target delayed by 7 bits, the ADALINE on channel 2 matched its channel 1 BER almost exactly. In other words, the "not linearly separable" problem in two dimensions largely disappears once the linear equaliser is given more taps and a decision delay.

### 2.7 Backpropagation (multi-layer perceptron)

A **multi-layer perceptron (MLP)** stacks layers of neurons, each computing a weighted sum and passing it through a smooth S-shaped function (here tanh), so the network as a whole can draw curved boundaries. **Backpropagation** trains it: compute the output error, work out how much each weight contributed to it by passing error signals backwards through the layers (the chain rule), and adjust every weight a little downhill. It is LMS generalised to several layers.

**Results:**
- Trained bit by bit (the way an equaliser would really have to work), it was **unreliable**: it failed to converge within 1,000 training bits at least once in every five trials, even on noiseless channels. This contradicted a 1989 paper claiming it out-adapted the linear equaliser, and this mode was dropped.
- As an experiment, **batch training** was tried instead: 40 noisy samples presented repeatedly for 1,000 epochs (an epoch is one pass over the set, with weights updated at its end), using the 2-6-6-1 structure suggested in the literature. This converged well: on channel 1 the error fell to about 10⁻² in about 10 epochs and about 10⁻⁴ by epoch 1,000; on channel 2 it sat on a plateau around 0.7 until about epoch 240, then dropped to about 6×10⁻⁴.
- But each trial took at least **ten times** the computation of an ADALINE trial; the learned boundaries differed from the Bayesian optimum and were different on every run, depending on the random starting weights. And batch training over repeated epochs is not practical for a live receiver anyway.

### 2.8 Recurrent network (RTRL)

A **recurrent neural network (RNN)** feeds its own outputs back, delayed by one symbol, as extra inputs, so it carries some memory of its past decisions — somewhat like a decision-feedback equaliser. The one used here had a single layer of output nodes, only one of which was the actual equaliser output.

It was trained by **Real-Time Recurrent Learning (RTRL)**. Because outputs feed back, each weight affects the output both directly and through earlier outputs. RTRL keeps, for every weight, a running "sensitivity" — how much each output would change if that weight changed — and updates it recursively every bit from its previous value. The weight is then nudged by error × sensitivity. This is more bookkeeping than backpropagation (sensitivities for every weight against every output).

**Results:**
- It converged well in noise on both toy channels, but was **very sensitive** to learning rate and network size; too high a rate made it go unstable quickly and, unlike backpropagation, it did not recover.
- Channel 1, 21 dB, 2 inputs, 3 output nodes, η = 0.005: error settled at about 0.055 from about bit 400 — noticeably worse than the ADALINE's 0.015.
- BER on channel 1: about 1.1×10⁻¹ at 6 dB, 4.5×10⁻² at 10 dB, 2×10⁻³ at 16 dB, 4×10⁻⁵ at 20 dB.

### 2.9 Radial-basis function network

An **RBF network** works differently. Each hidden node stores a reference point (a "centre") and fires strongly when the input is close to it, falling off as a Gaussian bell with distance; the width used was 0.125. For channel 1 with two inputs there are exactly 8 possible noise-free points, so the network had 8 RBF nodes, one per point. Their outputs go to a single linear output node whose weights are fixed at −1 for nodes that represent "−1 sent" and +1 for "+1 sent". Whichever node is nearest dominates, so the output lands near ±1.

Training is not gradient descent at all: since the desired symbols are known during the preamble, the network knows which node *should* fire for each received sample and simply averages those samples into that node's centre. Over time the noise averages out and each centre settles on its true point. That makes training cheap and quick (a short preamble suffices), and there is no error curve to plot — only BER. The cost moves to operation, where every bit needs a distance calculation for every node.

**Results:** it performed "extremely well" on both toy channels, with near-identical curves: channel 1 about 1.15×10⁻¹ at 6 dB, 4.5×10⁻² at 10 dB, 2.2×10⁻³ at 16 dB, about 4×10⁻⁵ at 20 dB; channel 2 slightly worse at low SNR and the same from 18 dB up.

### 2.10 The real comparison: the telephone-line channel

All four were then tried on the ten-coefficient telephone-line model, first without noise to find good settings, then across a range of SNRs.

- **Backpropagation:** no settings could be found that worked at a realistic network size. Not tested further.
- **RBF:** would not train at an acceptable size. The reason is structural: a 10-tap channel produces 2,048 possible received points, practically on top of one another. An RBF network needs a node per point, so it becomes enormous and too slow to be any use.
- **ADALINE:** 15 inputs, learning rate 0.08 as printed (the rebuild shows it must have been 0.008; 0.08 diverges), 5-bit decision delay — adapted successfully.
- **RNN:** 10 inputs, 4 output nodes, learning rate 0.003 — adapted successfully.

Convergence was very similar, the RNN finishing slightly lower (squared error about 0.5 against about 0.6–0.65 for the ADALINE). But on the measure that matters, bit errors, the ADALINE won, and the gap widened as noise fell:

| SNR (dB) | ADALINE BER | RNN BER |
|---|---|---|
| 6 | about 1.4×10⁻¹ | about 1.4×10⁻¹ |
| 10 | about 6×10⁻² | about 6×10⁻² |
| 14 | about 1.1×10⁻² | about 1.7×10⁻² |
| 16 | about 2.3×10⁻³ | about 6.5×10⁻³ |
| 18 | about 2.4×10⁻⁴ | about 2×10⁻³ |
| 20 | about 6×10⁻⁶ | about 3.5×10⁻⁴ |

(The original has no results table; these are read off its Figure 9.3.)

### 2.11 Conclusions

- **RBF and backpropagation are unsuitable** for this job: on a realistic channel the received points are too densely packed to separate with a network small enough to compete with the ADALINE on computation.
- **The RNN is usable but second best**: sensitive to starting parameters and prone to instability, and worse on BER. A contemporary paper (Bradley and Mars, 1994) went further and called RNNs unsuitable for equalisation.
- Overall, none of the three showed any advantage from the extra non-linear complexity, so the report concludes that ANNs are unsuitable for channel equalisation.
- The report is honest about its limits: it "only scratched the surface", other structures (Kohonen self-organising maps, FIR MLPs) were not tried, and the work was slowed by the computing time available. It suggests a deeper look at *why* the networks adapt poorly.

### 2.12 How the simulators were built

Everything was prototyped in MATLAB 4.0 with its Neural Network Toolbox, which proved far too slow (simulations running into tens of hours). The code was then ported to C — first as DLLs called from MATLAB, abandoned as unstable under Windows — and finally built as stand-alone DOS programs with Borland C/C++ 3.1.

The structure, as listed in Appendix B:

- **`EQ_FUNCS.C` / `.H`** — a shared library: the node, network and FIR-delay-line data structures, plus the common pieces — pseudo-random sequence generator, delay-line shifting, feed-forward through a layer (tanh, linear ADALINE, or RBF distance), backpropagation derivatives and weight updates, random weight initialisation, command-line parameter parsing, and saving results. Biases are handled as one extra node fixed at −1 in the layer above, rather than as separate fields (a revision dated April 1995).
- **One simulator per network**, each as a header, an `*_INIT.H` holding the globals and setup (allocate the network, read settings), and a main program running the train/test loop: `LMS.C` (ADALINE), `BPBC.C` (backpropagation), `RNN.C` (RTRL) and the RBF simulator (`RB.H`, `RB_INIT.H` and its main).
- **Text files for I/O.** Each program reads a constants file (e.g. `LMS_CON.TXT`, `RNN_CON.TXT`, `RB_CON.TXT`, `BP_CONST.TXT`) and optionally a `WEIGHTS.TXT` of starting weights, and writes `ERRORS.TXT` (the averaged error curve), `BER.TXT` and `FAIL.TXT` (count of failed trainings) for plotting back in MATLAB. The backpropagation program is the odd one out, with its file paths hard-coded rather than passed in.

*Author's note (2026 recollection, not in the original report):* the backpropagation results were collected by clearing the EE teaching lab at 17:00, booting each of the 16–20 bench PCs from a floppy holding the simulator, leaving them running overnight, and collecting the results at 07:00. That makes the hard-coded `e:\\temp` paths and the debug-style build less surprising: it was a program built for one job on lab machines, not a finished tool.

---

## 3. Addendum: "AI" in 1995 vs 2026

### Scale

The networks in this report were tiny. The ADALINE had 7–15 weights; the RBF on a toy channel had 8 centres; the 2-6-6-1 MLP has about 67 weights; the RNN a few dozen. They were trained one sample at a time on a single DOS PC, and running a 1,000-epoch batch experiment was a problem of "finding available computing time".

By 2026, mainstream AI means **deep learning**: networks with many layers, trained on GPUs or dedicated accelerators, in clusters of thousands of chips. Large language models built on the **transformer** architecture (2017) have parameter counts in the billions to, reportedly, trillions, and are trained on trillions of tokens of text. The gap from this report is something like nine to eleven orders of magnitude in parameters and far more in training compute.

### What is the same and what changed

The core mechanism is recognisably the same. Weighted sums, a non-linearity, an error measure, and gradient descent via the chain rule — backpropagation, exactly as in section 6 of the report — still train essentially every modern network. LMS itself is still everywhere in signal processing.

What changed is mostly the things that made the 1995 MLP misbehave:

- **Compute and data**, which turned "ten times slower than LMS" from a deal-breaker into a non-issue for many applications.
- **Training practice**: better weight initialisation, activation functions such as ReLU that avoid the flat regions where tanh networks stall (the long plateau in the channel 2 curve looks like that), adaptive optimisers such as Adam, normalisation layers, and mini-batches (a middle ground between the report's per-sample and full-batch modes). The unreliable convergence and run-to-run variation the report saw were real, widely shared problems in that era.
- **Recurrent training**: RTRL's cost grows steeply with network size, which is one reason it is rarely used; backpropagation through time, LSTMs and later transformers took over sequence modelling.

### Has the conclusion aged well?

**For its own question, largely yes.** For a slowly varying, essentially linear channel like a telephone line, with a requirement to adapt in real time from a short preamble on cheap hardware, a linear equaliser trained by LMS (or its relatives, plus decision feedback and Viterbi-style detection) was and remains the sensible engineering answer. Wired modems, and most of the equalisers in shipping radios today, are still built from classical adaptive filtering, not neural networks. The report's specific diagnoses also hold up: RBF/Bayesian-style equalisers do scale badly, because the number of possible received patterns grows exponentially with channel length (the 2,048 points of section 9); and the comparison used the fair yardstick of BER at equal computational budget, which is still the right test.

**The one-line summary ("ANNs are unsuitable for channel equalisation") reads more broadly than the report's conclusions, which were explicitly about practically sized networks at 1995 computing cost.** On that narrower question the answer has largely stayed true for simple linear channels. Where things have moved is exactly where the report pointed: since the late 2010s there has been substantial research on machine learning in the physical layer — learned channel estimation and equalisation, "neural receivers" that replace several receiver blocks with one trained network, and end-to-end learned links. Vendors and research groups have demonstrated neural receivers for 5G-style signals, and AI/ML for the air interface is a 3GPP study topic and a prominent 6G theme. These help most in conditions the 1995 tests did not cover — strongly non-linear impairments, complex multi-antenna channels, hard-to-model hardware — and they are trained offline on large datasets with compute that would have been unthinkable in a 1995 modem. How much of this is in shipping products is still evolving; classical methods remain the default in most deployed links.

**None of that is the problem the report set.** The project was about *adaptive* equalisers: ones that learn the channel they are on, from that channel's own signal. Each call's line is different, the preamble has a maximum acceptable length, and the learning has to fit a modem's computation budget. A canned network trained elsewhere across many channels is a different design, and it needs training resources on a scale nobody had in 1995; a lab of 486s running overnight could barely train the report's small networks. The project did lean towards offline training in one place: the backpropagation program trains for 1000 epochs over a stored set of preamble patterns, which could never run in real time. But it was still learning the channel in use, which is exactly the gap the "practically sized network" conclusion measures. The 2026 re-run (`report_2026.md`) tests the adaptive case, learning from a short preamble, and RLS and the decision-feedback equaliser still beat every network.

So the fair reading is: the report was right that, with 1995 networks, training methods and compute, and for a linear telephone channel learned afresh from a bounded preamble, neural networks offered no advantage over LMS, and under those same constraints that is still true in 2026. Its closing caveat — that better structures or algorithms might change the picture — has come true, but for a different problem: trained rather than adaptive equalisers, built with training resources on a scale that did not exist in 1995.
