"""2026 re-run: figures and tables from results/y2026/*.csv -> results/y2026/figures/, results/y2026/tables.md.

    uv run scenarios/plot_y2026.py [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import csv  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results" / "y2026"
SUF = "_quick" if "--quick" in sys.argv else ""
ORDER = ["LMS", "RLS", "DFE", "RBF", "MLP", "CNN", "TF", "GRU", "MAP", "MLSE"]
STYLE = {"LMS": ("#7f7f7f", "o", "-"), "RLS": ("#404040", "s", "-"), "DFE": ("#8c564b", "D", "-"),
         "RBF": ("#9467bd", "v", "-"), "MLP": ("#1f77b4", "^", "-"), "CNN": ("#2ca02c", "<", "-"),
         "TF": ("#ff7f0e", ">", "-"), "GRU": ("#d62728", "P", "-"), "MAP": ("black", "", "--"),
         "MLSE": ("black", "", ":")}
TITLES = {"ch1": "Channel 1: 1 + 0.5z$^{-1}$", "ch2": "Channel 2: 0.5 + z$^{-1}$",
          "ch3": "Channel 3: 10-tap telephone", "ch1nl": "Channel 1 + nonlinearity"}


def load(name):
    with open(RES / f"{name}{SUF}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def ber_plot(rows):
    chans = [c for c in ["ch1", "ch2", "ch1nl", "ch3"] if any(r["channel"] == c for r in rows)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, ch in zip(axs.flat, chans):
        for mth in ORDER:
            rr = sorted([r for r in rows if r["channel"] == ch and r["method"] == mth],
                        key=lambda r: r["snr_db"])
            if not rr:
                continue
            x = np.array([r["snr_db"] for r in rr])
            y = np.array([r["ber"] for r in rr])
            lo = np.array([r["cp_lo"] for r in rr])
            hi = np.array([r["cp_hi"] for r in rr])
            nz = y > 0
            c, mk, ls = STYLE[mth]
            ax.errorbar(x[nz], y[nz], yerr=[y[nz] - lo[nz], hi[nz] - y[nz]], color=c, marker=mk,
                        ls=ls, ms=4, lw=1.2, capsize=2, label=mth)
            if (~nz).any():  # zero errors: plot the 95% upper bound as an open marker
                ax.plot(x[~nz], hi[~nz], ls="none", marker=mk or "x", mfc="none", color=c)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.5)
        ax.set_title(TITLES[ch])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
    axs.flat[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Offline-trained equalisers (95% Clopper-Pearson CIs; open = 0 errors, upper bound)")
    fig.tight_layout()
    fig.savefig(RES / "figures" / f"ber_offline{SUF}.png", dpi=130)


def preamble_plot(rows):
    chans = [c for c in ["ch1", "ch2", "ch1nl", "ch3"] if any(r["channel"] == c for r in rows)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, ch in zip(axs.flat, chans):
        for mth in ORDER:
            rr = [r for r in rows if r["channel"] == ch and r["method"] == mth]
            if not rr:
                continue
            ns = sorted({r["n_train"] for r in rr})
            k = [sum(r["errors"] for r in rr if r["n_train"] == n) for n in ns]
            b = [sum(r["bits"] for r in rr if r["n_train"] == n) for n in ns]
            med = [np.median([r["ber"] for r in rr if r["n_train"] == n]) for n in ns]
            c, mk, ls = STYLE[mth]
            ax.plot(ns, np.maximum(np.array(k) / np.array(b), 1e-6), color=c, marker=mk, ls=ls,
                    label=mth)
            del med
        snr = rows[[r["channel"] for r in rows].index(ch)]["snr_db"]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{TITLES[ch]}, {snr:.0f} dB")
        ax.set_xlabel("preamble length N (symbols)")
        ax.set_ylabel("BER (pooled over trials)")
        ax.grid(True, which="both", alpha=0.3)
    axs.flat[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Adaptation from a short PN preamble (1995 deployment constraint)")
    fig.tight_layout()
    fig.savefig(RES / "figures" / f"ber_preamble{SUF}.png", dpi=130)


def tables(off, pre):
    out = []
    out.append("## Offline: BER at selected SNR (errors/bits)\n")
    pick = {"ch1": 10, "ch2": 10, "ch1nl": 10, "ch3": 16}
    out.append("| method | " + " | ".join(f"{c} @ {s} dB" for c, s in pick.items()) +
               " | params (ch1/ch3) | MACs/sym (ch1/ch3) | train s (ch1 @10) |")
    out.append("|---" * (len(pick) + 4) + "|")
    for mth in ORDER:
        cells, pc, mc = [], [], []
        tt = ""
        for c, s in pick.items():
            r = [r for r in off if r["channel"] == c and r["method"] == mth and r["snr_db"] == s]
            if not r:
                cells.append("–")
                continue
            r = r[0]
            cells.append(f"{r['ber']:.2e}" if r["errors"] else f"<{r['cp_hi']:.1e}")
            if c in ("ch1", "ch3"):
                pc.append(f"{int(r['params'])}")
                mc.append(f"{int(r['macs'])}")
            if c == "ch1" and isinstance(r.get("train_time"), float):
                tt = f"{r['train_time']:.1f}"
        out.append(f"| {mth} | " + " | ".join(cells) + f" | {'/'.join(pc)} | {'/'.join(mc)} | {tt} |")
    out.append("\n## Preamble: pooled BER vs preamble length N\n")
    for ch in ["ch1", "ch2", "ch1nl", "ch3"]:
        rr = [r for r in pre if r["channel"] == ch]
        if not rr:
            continue
        ns = sorted({int(r["n_train"]) for r in rr})
        out.append(f"\n**{ch} @ {rr[0]['snr_db']:.0f} dB**\n")
        out.append("| method | " + " | ".join(f"N={n}" for n in ns) + " |")
        out.append("|---" * (len(ns) + 1) + "|")
        for mth in ORDER:
            q = [r for r in rr if r["method"] == mth]
            if not q:
                continue
            cells = []
            for n in ns:
                z = [r for r in q if r["n_train"] == n]
                cells.append(f"{sum(r['errors'] for r in z) / sum(r['bits'] for r in z):.1e}"
                             if z else "–")
            out.append(f"| {mth} | " + " | ".join(cells) + " |")
    (RES / f"tables{SUF}.md").write_text("\n".join(out) + "\n")


def main():
    (RES / "figures").mkdir(exist_ok=True)
    off, pre = load("offline"), load("preamble")
    ber_plot(off)
    preamble_plot(pre)
    tables(off, pre)


if __name__ == "__main__":
    main()
