"""Comparison plots: every implementation vs the values read off the 1995 figures.
uv run --no-project --with numpy --with matplotlib python scenarios/plot.py
Reads results/rebuild/*.csv and curves/, writes results/rebuild/*.png.
--out DIR (e.g. fixed, relative to results/; default rebuild) reads and writes that directory instead.
"""
import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / (sys.argv[sys.argv.index("--out") + 1]
                                                 if "--out" in sys.argv else "rebuild")
# Values read off the original figures (approximate; docs/report_text.md, docs/report_rewrite.md)
ORIG_BER = {
    "fig5_5": {6: 4e-2, 10: 3.5e-3, 12: 3e-4, 14: 7e-6},
    "fig7_3": {6: 1.1e-1, 8: 7.5e-2, 10: 4.5e-2, 12: 2.2e-2, 14: 8e-3, 16: 2e-3, 18: 3.5e-4, 20: 4e-5},
    "fig8_3_ch1": {6: 1.15e-1, 8: 8e-2, 10: 4.5e-2, 12: 2.2e-2, 14: 8.5e-3, 16: 2.2e-3, 18: 3.8e-4, 20: 4e-5},
    "fig8_3_ch2": {6: 1.35e-1, 8: 9.5e-2, 10: 5.5e-2, 12: 2.7e-2, 14: 1.1e-2, 16: 3e-3, 18: 3.8e-4, 20: 4e-5},
    "fig9_3_ada": {6: 1.4e-1, 10: 6e-2, 14: 1.1e-2, 16: 2.3e-3, 18: 2.4e-4, 20: 6e-6},
    "fig9_3_ada008": {6: 1.4e-1, 10: 6e-2, 14: 1.1e-2, 16: 2.3e-3, 18: 2.4e-4, 20: 6e-6},
    "fig9_3_rnn": {6: 1.4e-1, 10: 6e-2, 14: 1.7e-2, 16: 6.5e-3, 18: 2e-3, 20: 3.5e-4},
}
STYLE = {"dos": ("o", "C0"), "clang": ("s", "C1"), "pyline": ("x", "C2"), "numpy": ("^", "C3"),
         "clang_fix": ("v", "C4"), "numpy_fix": ("<", "C5"), "clang_20": ("d", "C6"),
         "pyfix": ("P", "C7"), "pyfix_pn31": ("*", "C8")}


def load(name):
    p = OUT / f"{name}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def ber_plot(name, title):
    rows = load(name)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for impl in dict.fromkeys(r["impl"] for r in rows):
        pts = [(float(r["setting"].split("=")[1]), float(r["ber"])) for r in rows
               if r["impl"] == impl and r["ber"] not in ("", "None")]
        pts = [(s, b if b > 0 else 3e-7) for s, b in pts]
        m, c = STYLE.get(impl, (".", "k"))
        ax.semilogy(*zip(*pts), marker=m, color=c, lw=1, label=impl, alpha=0.8)
    if name in ORIG_BER:
        o = ORIG_BER[name]
        ax.semilogy(list(o), list(o.values()), "k--", marker="*", ms=10, label="1995 report (read off figure)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER (0 plotted at 3e-7)"); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=110); plt.close(fig)


def curve_plot(name, title, impls=("dos", "clang", "pyline", "numpy", "pyfix", "pyfix_pn31")):
    files = sorted((OUT / "curves").glob(f"{name}__*.txt"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    settings = list(dict.fromkeys(f.stem.split("__")[2] for f in files))
    for k, s in enumerate(settings):
        for impl, ls in zip(impls, ("-", "--", ":", "-.", "-", "--")):
            f = OUT / "curves" / f"{name}__{impl}__{s}.txt"
            if f.exists():
                y = [float(v) for v in f.read_text().split()]
                ax.semilogy(y, ls, color=f"C{k}", lw=0.8, label=f"{s} {impl}")
    ax.set_xlabel("preamble bit"); ax.set_ylabel("ensemble squared error"); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_curves.png", dpi=110); plt.close(fig)


ber_plot("fig5_5", "Fig 5.5: ADALINE 7 taps, channel 1")
ber_plot("fig5_7", "Fig 5.7: ADALINE 7 taps, channel 2, delay 7")
ber_plot("fig7_3", "Fig 7.3: RNN 2 in / 3 nodes, channel 1")
ber_plot("fig8_3_ch1", "Fig 8.3: RBF 8 nodes, channel 1")
ber_plot("fig8_3_ch2", "Fig 8.3: RBF 8 nodes, channel 2")
ber_plot("fig9_3_ada", "Fig 9.3: ADALINE 15 taps, telephone channel")
ber_plot("fig9_3_ada008", "Fig 9.3: ADALINE 15 taps, eta=0.008 (report says 0.08)")
ber_plot("fig9_3_rnn", "Fig 9.3: RNN 10 in / 4 nodes, telephone channel")
ber_plot("extra_ada10_real", "Extra: ADALINE 10 taps, telephone channel (RNN input size)")
ber_plot("bp_ch1", "BPBC batch backprop 2-6-6-1, channel 1 (lr/momentum assumed)")
ber_plot("bp_ch2", "BPBC batch backprop 2-6-6-1, channel 2 (lr/momentum assumed)")
ber_plot("extra_ada2", "Extra: ADALINE 2 taps, channel 1 (RNN/RBF input size)")
curve_plot("fig5_3", "Fig 5.3: ADALINE taps 1/2/3/7/11, ch1, 21 dB")
curve_plot("fig5_4", "Fig 5.4: ADALINE eta, 7 taps, ch1, 21 dB")
curve_plot("fig9_3_ada008", "Fig 9.2: ADALINE telephone channel, eta=0.008")
curve_plot("fig9_3_rnn", "Fig 9.2: RNN telephone channel", impls=("clang", "clang_fix", "dos", "pyline", "pyfix", "pyfix_pn31"))
curve_plot("fig7_2", "Fig 7.2: RNN convergence, ch1, 21 dB", impls=("clang", "clang_fix", "dos", "pyline", "pyfix", "pyfix_pn31"))


def overview():
    """One chart answering the main question: telephone channel, ADALINE vs RNN, as built and fixed."""
    def series(name, impl):
        rows = [r for r in load(name) if r["impl"] == impl and r["ber"] not in ("", "None")]
        pts = sorted((float(r["setting"].split("=")[1]), max(float(r["ber"]), 3e-7)) for r in rows)
        return zip(*pts) if pts else ([], [])
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    if OUT.name != "fixed":
        ax.semilogy(*series("fig9_3_ada008", "clang"), "o-", color="C0", label="ADALINE 15 taps (1995 code, eta 0.008)")
        ax.semilogy(*series("extra_ada10_real", "clang"), "o--", color="C9", label="ADALINE 10 taps (same inputs as RNN)")
        ax.semilogy(*series("fig9_3_rnn", "clang"), "s-", color="C3", label="RNN as written (feedback never used)")
        ax.semilogy(*series("fig9_3_rnn", "clang_fix_RNN-FFWD"), "s--", color="C1", label="RNN with feedback fixed")
    else:
        ax.semilogy(*series("fig9_3_ada008", "pyfix"), "o-", color="C0", label="ADALINE 15 taps, all fixes (eta 0.008)")
        ax.semilogy(*series("extra_ada10_real", "pyfix"), "o--", color="C9", label="ADALINE 10 taps, all fixes")
        ax.semilogy(*series("fig9_3_rnn", "pyfix"), "s-", color="C3", label="RNN, all fixes (eta re-tuned)")
        ax.semilogy(*series("fig9_3_rnn", "pyfix_pn31"), "s--", color="C1", label="RNN, all fixes, 31-bit PN preamble")
    o = ORIG_BER["fig9_3_ada"]; ax.semilogy(list(o), list(o.values()), "k*:", ms=9, label="1995 report ADALINE (read off)")
    o = ORIG_BER["fig9_3_rnn"]; ax.semilogy(list(o), list(o.values()), "kx:", ms=8, label="1995 report RNN (read off)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("bit error rate"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Telephone channel: rebuilt 1995 programs, 100 trials"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "overview_telephone.png", dpi=120); plt.close(fig)


overview()
