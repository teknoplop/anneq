# FPGA backprop engine on the Icepi Zero (#22)

The question: how much faster can an Icepi Zero (Lattice LFE5U-25F, 24k LUTs, 28 18x18 multipliers,
FT231X USB UART) train the 1995 BPBC network? BPBC is a 2-6-6-1 tanh MLP trained by batch
backprop with momentum, on 40 patterns for 1000 epochs, with lr 0.1 and momentum 0.5.

**Short answer.** The board runs about 2.1 M pattern presentations/s on chip (1.4 M/s end to end
over USB). That is roughly 500x a 486. But it is 3-4x *slower* than one M4 core running the C
(5-12x slower against the fixed-point C), and about 1.7x slower than numpy. The network is too
small and too serial for a 25k-LUT FPGA at 50 MHz to beat a modern CPU core.

## Design (`rtl/bp_engine.v`, `rtl/top.v`)

- **One engine trains one trial.** Each engine has one pipelined 18x18 MAC (M1) and a second
  multiplier (M3) for `der*x` and `h^2`. The epoch update `dl = (g*K1 + dl*K2)>>16` runs on M1
  in five 17-bit chunks.
- **Number formats.** Activations and deltas are Q3.14. Weights are 32-bit Q3.28 masters, used
  as Q3.14 in the multiplies. Gradient sums are 36-bit. tanh comes from a 2048-entry LUT over
  |x|<4, and `1-h^2` is stored when each activation is written.
- **Phases.** Each sample runs through 9 phases (3 forward, 3 backward, 3 gradient). A 5-stage
  pipeline drains between dependent phases. The cost is 233 cycles per presentation, of which 177
  are MAC issues (76% MAC utilisation), including the epoch update.
- **Memory.** Per engine: the training set (up to 63 patterns) in LUTRAM; weights and momentum in
  one EBR; gradients in one EBR; the tanh LUT in two EBRs.
- **Top level.** 10 engines in parallel behind a 2 Mbaud UART. The commands write weights, data
  and parameters, start training, read weights and return the cycle count. Only these cross
  the USB link.
- **Verification.** `ref/bpfix.c` is a bit-exact C model. The verilator testbench
  (`sim/tb_engine.cpp`) and the board both reproduce its final weights bit for bit.

## Implementation (yosys + nextpnr-ecp5, `../results/fpga/utilisation.txt`)

10 engines: 21 954 / 24 288 LUTs (90%), 10 595 FFs (43%), 40/56 EBR, 20/28 MULT18X18D.
Fmax is 54.3 MHz, so it runs from the board's 50 MHz oscillator. 11 engines fit at 97% (52.8 MHz).
12 or more do not fit, because of logic (the activation and delta register-file muxes), not multipliers.

## Measured (Mac mini M4, 20 trials x 1000 epochs x 40 patterns = 800 000 presentations)

| Implementation | Time | Presentations/s | FPGA speed-up |
|---|---|---|---|
| 486, estimate (about 2 MFLOPS, about 640 flop-equivalents per presentation incl. 13 tanh) | about 4.3 min | about 3e3 | **about 700x on chip, about 450x end to end** |
| BPBC.C faithful (`src/clang` build_fix), 1 core | 0.11 s | 7.3e6 | 0.29x |
| `ref/bpfix.c` double, 1 core | 0.082 s | 9.7e6 | 0.22x |
| `ref/bpfix.c` fixed point, 1 core | 0.032 s | 2.5e7 | 0.09x |
| numpy batch (all trials x patterns vectorised) | 0.21 s | 3.7e6 | 0.58x |
| **FPGA, 10 engines, on chip** | 0.372 s | **2.15e6** | |
| **FPGA, end to end incl. USB** | 0.568 s | **1.41e6** | |

Per engine: 50 MHz / 233 cycles is 215 k presentations/s, about 1/34 of one M4 core in double.
The M4 has 10 cores, so with the 20 trials spread across cores the CPU lead grows about 10x.
ADALINE/LMS (7 taps) runs at 4.2e7 updates/s natively and 9.1e6/s in numpy. It is far too small
to be worth putting on an FPGA, so it was not.

## Fixed-point accuracy cost (ch1, 20 trials, BER over 10 000 bits, float forward pass)

| SNR dB | numpy float64 | fixed point (= board) mean / median |
|---|---|---|
| 6 | 5.70e-2 | 5.93e-2 / 5.41e-2 |
| 10 | 1.08e-2 | 1.33e-2 / 1.09e-2 |
| 14 | 8.75e-4 | 2.08e-3 / 9.0e-4 |
| 21 | 0 | 2.0e-5 / 0 |

The medians match float. The means are pulled up by one or two trials per SNR that settle
somewhere worse. Training MSE is the same or lower (2.7e-6 vs 2.6e-6 at 14 dB).

## What limits it

1. **Serial dependencies within one small network.** Forward, backward and gradient phases depend
   on each other, and 67 weights give little parallelism, so each engine is one MAC per cycle.
2. **Logic (LUTs) caps it at 10-11 engines.** Multipliers (20/28) and EBR (40/56) are not the cap.
   Even with all 28 multipliers busy every cycle at 50 MHz, about 200 MACs per presentation gives
   at most about 7 M/s, still below one M4 core running the fixed-point C.
3. **USB** takes 34% of end-to-end time (about 0.19 s per 20 trials, mostly loading data and
   weights at 2 Mbaud). The cost is per trial, so longer training would hide it.
4. The clock (50 MHz, Fmax 54) is a minor factor. A PLL could add 10-20% at most.

Reproduce: `gateware/run_mini.sh` (see header). The Icepi Zero pinout is from `vendor/icepi-zero`
(gateware/uart/icepi-zero.lpf).
