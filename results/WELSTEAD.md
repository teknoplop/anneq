# 1995 code vs Welstead's book disk (#21)

Question: is any of the 1995 equaliser code (src/borland/) copied or adapted from the
companion disk of S. T. Welstead, *Neural Network and Fuzzy Logic Applications in
C/C++* (Wiley, 1994)?

**Answer: no. All 14 files are independent.** The two code bases share no code,
no distinctive identifiers, no comments and no data structures. Where they
implement the same textbook idea (backprop with momentum) they do it differently.
None of the bugs #10-#19 can come from the disk.

The disk itself is not in this repository (Wiley licence: single-computer copy
only). Nothing below reproduces its code beyond a few identifiers.

## How the disk was read

- 1.44 MB image, all sectors read. It holds a Wiley installer (LZEXE-packed
  INSTALL.EXE, INSTALL.RES) and one compressed archive VOLUME.001 with a
  proprietary header ("WELSTEAD Disk 1"). Raw-deflate and PKWARE-DCL probes at
  the data start found nothing, so no existing unpacker applied.
- The installer was run headless in DOSBox-X (local). It is a full-screen menu
  that writes straight to video memory, and DOSBox-X autotype arrow keys did not
  reach it, so a 150-byte INT 16h TSR (assembled with nasm) fed it the keys
  Down, Down, Down, Enter ("Start installation", default C:\NN) and dumped the
  text screen on each key read so the menus could be followed.
- Result: 165 files, 2,656,328 bytes, matching the installer's own total, in
  the 17 directories listed in README.DOC. Kept only in the git-excluded
  disk folder.

## What the disk contains (relevant parts)

FF (feedforward net, one hidden layer), FFADV (conjugate gradient, simulated
annealing, GA training), TRN (training-file I/O), UTC (vectors/matrices, file
helpers, random numbers). Applications: pattern classification, chaotic
dynamics, financial series, broom balancing, Hopfield, Kohonen, fuzzy control.
There is **no** ADALINE/LMS, no RBF network, no recurrent network or RTRL, no
tanh activation, no Gaussian noise generator, no FIR channel model and no PN
sequence anywhere on the disk.

## Token-level similarity

Winnowing fingerprints (12-token k-grams, window 8, comments and preprocessor
lines removed), each 1995 file against all ~120 C/C++/H files on the disk:

| measure | 1995 vs disk (best match per file) | disk vs disk, different directories |
|---|---|---|
| raw tokens | 0.000-0.021 | - |
| identifiers normalised | 0.000-0.115 (RB.C vs GA/GAPROCS.C) | median 0.012, p90 0.063 |

The normalised scores are what any two pieces of 1990s C sharing `for` loops
score; the best match for each file is a different, unrelated disk directory
(GA, FFIN, PTTRN, FL), not FF. The longest common run of raw tokens between
any 1995 file and any disk file, after dropping `#include` lines, is 14 tokens:
a pair of nested `for (i = 0; i < ...` loop headers.

Shared identifiers (length >= 4, minus C keywords and the C library) are
English words (`weights`, `layer`, `momentum`, `training`), Borland conio
calls (`_setcursortype`, `wherey`, `textmode`), and one function name,
`print_weights`, which on the disk writes a weights file with a different
signature while in EQ_FUNCS.C it prints a weight matrix to the screen.

## Algorithm comparison

| aspect | 1995 (EQ_FUNCS, BPBC, BP_INIT) | Welstead disk (FF, UTC) |
|---|---|---|
| data layout | linked node structs (`NODE_STR` with per-node `WEIGHT_STR` array), 0-based | 2-D weight matrices `wi`/`wo`, 1-based vectors |
| layers | up to 3 weight layers (BPBC uses 2 hidden) | exactly one hidden layer |
| activation | bipolar `tanh` (SIGMOID macro), derivative `1-y^2` | logistic `1/(1+exp(-x))` clamped at +-50, derivative `y(1-y)`; linear output scaled by 0.1 |
| bias | extra node in the layer below (v2, 08/04/95) | constant threshold input `x[n+1]=1` |
| training | batch: gradients summed over the epoch, applied once per epoch | online: update after each pattern, only if its error exceeds a threshold, optional shuffling |
| momentum | `dw = lr(1-mc)g + mc*dw_prev` (stored step includes lr) | `w += lr((1-a)g + a*g_prev)` (stored term is the raw gradient) |
| weight init | `rand()/RAND_MAX * 0.1`, all positive, in `[0, 0.1]` | `range * (2*frand()-1)`, symmetric |
| RNG | `srand(time)` every trial, `rand()/RAND_MAX` (can return 1.0) | `randomize()` once, `rand()/(RAND_MAX+1.0)` |
| noise | Box-Muller Gaussian scaled by SNR | none |
| file I/O | fixed-name text files read with fscanf, `save_doubles` writes one number per line | menu-driven Turbo Vision setup, named .SET/.TRN/.WTS files with headers |
| UI | conio banner + `gotoxy` status line | Turbo Vision dialogs |

The `(1 - mc)` momentum form is shared in spirit but is the common textbook
form (it is also MATLAB's `traingdm`); the two implementations differ in what
is stored and where the learning rate is applied.

## Verdicts per file

| file | verdict | evidence |
|---|---|---|
| EQ_FUNCS.C / .H | independent | tanh, Box-Muller, FIR, RBF, PN, node-struct layout: none on disk; backprop written differently (batch vs online, different momentum storage) |
| BPBC.C / .H, BP_INIT.H | independent | 3-layer batch BP over an equaliser preamble; disk FF is 1 hidden layer, online, logistic |
| LMS.C / .H, LMS_INIT.H | independent | no ADALINE/LMS on disk |
| RNN.C, RNNBC.H, RNN_INIT.H | independent | no recurrent net or RTRL on disk |
| RB.C / .H, RB_INIT.H | independent | no RBF on disk |

No file qualifies even as "inspired by": the only common algorithm is generic
backprop with momentum, and the 1995 version follows a different formulation.

## Bugs #10-#19

| issue | inherited from Welstead? |
|---|---|
| #10 RNN feedback never connected | no: no RNN on disk |
| #11 RTRL term accumulated nodes_1 times | no: no RTRL on disk |
| #12 fill_rand heap overrun | no: disk initialises 2-D matrices sized from the setup, no `fill_rand` |
| #13 ADALINE bias never set | no: disk sets its threshold input on every feedforward |
| #14 unbounded retrain loop | no: disk has no retry-on-failure loop |
| #15 srand(time) per trial | no: disk seeds once with `randomize()` |
| #16 repeated 31-bit PN preamble | no: no PN generator on disk |
| #17 BPBC debug build | no: disk `print_weights` is a different function; paths and key waits are 1995-specific |
| #18 void main, argv[0], for(i=i) | no |
| #19 toy-channel figures differ | not explained by the disk (no equaliser code on it) |

## Conclusion for publishing

The 1995 sources contain no Welstead code, so the book's licence places no
restriction on them. The repository can go public again as long as the disk
image and its unpacked files stay out of git (they are excluded via
`.git/info/exclude` and were never committed).
