"""Impact of issue #10 (RNN feedback never wired) on the telephone channel, Fig 9.3 setting.

Writes results/issue10.png and results/other_nets.png from results/rebuild/ and results/fixed/.
Usage: uv run --no-project --with matplotlib python scenarios/plot_issue10.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"


def series(path, impl):
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["impl"] == impl and float(row["ber"]) > 0:
                pts.append((float(row["setting"].split("=")[1]), float(row["ber"])))
    return zip(*sorted(pts))


lines = [
    (R / "rebuild/fig9_3_rnn.csv", "clang", "RNN as in 1995 (feedback not wired)", "C3", "o-"),
    (R / "rebuild/fig9_3_rnn.csv", "clang_fix_RNN-FFWD", "RNN with #10 fixed only (1995 learning rate)", "C1", "s--"),
    (R / "fixed/fig9_3_rnn.csv", "pyfix", "RNN with all fixes, learning rate re-tuned", "C2", "^-"),
    (R / "fixed/fig9_3_ada008.csv", "pyfix", "ADALINE (LMS), fixed, η = 0.008", "C0", "d:"),
]
fig, ax = plt.subplots(figsize=(7, 4.5))
for path, impl, label, colour, style in lines:
    snr, ber = series(path, impl)
    ax.semilogy(snr, ber, style, color=colour, label=label)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Bit error rate")
ax.set_title("Telephone channel (Fig 9.3 setting): impact of issue #10")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "issue10.png", dpi=150)
print("wrote", R / "issue10.png")


# The other two networks: neither had a bug in its training maths, so the fixes barely move them.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
for path, impl, label, colour, style in [
    (R / "rebuild/fig8_3_ch1.csv", "clang", "RBF ch1, as in 1995", "C3", "o-"),
    (R / "fixed/fig8_3_ch1.csv", "pyfix", "RBF ch1, all fixes", "C2", "^--"),
    (R / "rebuild/fig8_3_ch2.csv", "clang", "RBF ch2, as in 1995", "C1", "s-"),
    (R / "fixed/fig8_3_ch2.csv", "pyfix", "RBF ch2, all fixes", "C0", "d--"),
]:
    snr, ber = series(path, impl)
    a1.semilogy(snr, ber, style, color=colour, label=label)
a1.set_title("RBF (Fig 8.3): fixes make no difference")
for path, impl, label, colour, style in [
    (R / "rebuild/bp_ch1.csv", "clang_fix", "Backprop ch1, as in 1995 (debug dumps removed)", "C3", "o-"),
    (R / "fixed/bp_ch1.csv", "pyfix_pn31", "Backprop ch1, all fixes, 31-bit PN preamble", "C2", "^--"),
    (R / "fixed/bp_ch1.csv", "pyfix", "Backprop ch1, all fixes, random preamble", "C0", "d:"),
]:
    snr, ber = series(path, impl)
    a2.semilogy(snr, ber, style, color=colour, label=label)
a2.set_title("Backprop (Fig 6.2 setting), 20 trials")
for ax in (a1, a2):
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit error rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "other_nets.png", dpi=150)
print("wrote", R / "other_nets.png")
