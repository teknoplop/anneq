"""Regenerate the results figures of the 1995 report from the rebuilt programs' output.

uv run --no-project --with numpy --with matplotlib python docs/report_1995/plots/make_plots.py \
    [--data results/rebuild] [--impl clang] [--out docs/report_1995/plots]

--data  directory holding <scenario>.csv and curves/<scenario>__<impl>__<setting>.txt
        (scenarios/run_scenarios.py output). Point it at a corrected run to redo the figures.
--impl  which implementation's results to plot (default clang = the 1995 code built unfixed;
        pyline gives the same numbers). If a setting is missing for --impl (hung run), the
        --fallback implementations are tried in order and the substitution is printed and
        written to <out>/substitutions.txt so the report footnotes can say so.
Figures computed analytically (Gaussian noise histogram, Bayes boundaries, sigmoid, impulse
response) need no data.
"""
import argparse, csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
here = pathlib.Path(__file__).resolve().parent
ap.add_argument("--data", default=str(here.parents[2] / "results" / "rebuild"))
ap.add_argument("--impl", default="clang")
ap.add_argument("--fallback", default="pyline,numpy")
ap.add_argument("--out", default=str(here))
a = ap.parse_args()
DATA, OUT = pathlib.Path(a.data), pathlib.Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
IMPLS = [a.impl] + [i for i in a.fallback.split(",") if i]
subs = []
plt.rcParams.update({"font.size": 9, "figure.dpi": 100})
REAL = [0.047309, 0.320619, 0.782586, 0.426974, 0.293453, 0.100099, 0.040872, 0.037772, 0.020498, 0.010807]


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)


def ber(scn):
    p = DATA / f"{scn}.csv"
    rows = list(csv.DictReader(open(p))) if p.exists() else []
    pts = {}
    for impl in IMPLS:
        for r in rows:
            if r["impl"] == impl and r["ber"] not in ("", "None"):
                s = float(r["setting"].split("=")[1])
                if s not in pts:
                    pts[s] = float(r["ber"])
                    if impl != a.impl:
                        subs.append(f"{scn} {r['setting']}: {impl} (no {a.impl} result)")
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def curve(scn, setting, all_trials=False):
    # all_trials: prefer the <impl>_all curve (every trial's last attempt) where the run wrote one;
    # the corrected (capped-retry) runs do so when some trials never pass the training test.
    for impl in ([a.impl + "_all"] if all_trials else []) + IMPLS + ["clang_fix"]:
        f = DATA / "curves" / f"{scn}__{impl}__{setting}.txt"
        if f.exists():
            if impl not in (a.impl, a.impl + "_all"):
                subs.append(f"{scn} {setting} curve: {impl} (no {a.impl} curve)")
            return np.array([float(v) for v in f.read_text().split()])
    subs.append(f"{scn} {setting} curve: missing"); return None


def lo_decade(y, lo):
    # lower the axis limit by whole decades if a curve goes below it (corrected RNN runs)
    m = np.nanmin(y[y > 0]) if y is not None and (y > 0).any() else lo
    return min(lo, 10 ** np.floor(np.log10(m)))


def ber_axes(ax, xlab="SNR (dB)", ylab="BER", lo=1e-6, hi=1e-1, xl=(5, 20)):
    ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_xlim(*xl)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(True, which="both", alpha=0.3)


def plot_ber(ax, xs, ys, **kw):
    # a BER of 0 (no errors in 100 x 10,000 bits) cannot be shown on a log axis: drop it
    pts = [(x, y) for x, y in zip(xs, ys) if y > 0]
    if pts:
        ax.plot(*zip(*pts), **kw)


# ---- Fig 1.1 / 2.2: illustrative received pulse (sketches in the original)
def pulse(t):  # rises in bit 0, peaks at boundary, decays into bit 2
    return np.where(t < 0, 0, (t / 1.0) ** 2 * np.exp(2 * (1 - t)))
t = np.linspace(0, 3, 400)
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot([0, 0, 1, 1, 3], [0, 1, 1, 0, 0], "k-", label="Transmitted Symbol")
ax.plot(t, pulse(t), "k--", label="Recieved Signal")
for b in (1, 2): ax.axvline(b, color="0.6", lw=0.6)
for i in range(3): ax.text(i + 0.5, -0.15, f"Bit {i}", ha="center")
ax.set_ylim(-0.25, 1.3); ax.set_xlabel("t"); ax.set_ylabel("Signal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=7, loc="upper right"); save(fig, "fig1_1")
fig, ax = plt.subplots(figsize=(5, 2.6))
tt = np.linspace(0, 4, 400); x = pulse(tt - 0.5 + 0.5)
ax.plot(tt, pulse(tt), "k--", label="Recieved Analogue Signal")
for k in (1, 2, 3): ax.vlines(k - 0.5 + 0.5, 0, pulse(np.array(k * 1.0)), "k", lw=2)
ax.vlines([], 0, 0, "k", lw=2, label="Signal Sample Levels")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["T", "2T", "3T"]); ax.set_yticks([])
ax.set_xlabel("t"); ax.set_ylabel("x(nT)"); ax.legend(fontsize=7); save(fig, "fig2_2")

# ---- Fig 2.3: 8192 samples of eq (2-4), mean 0, variance 0.01
rng = np.random.default_rng(1995)
u1, u2 = rng.random(8192), 1 - rng.random(8192)
g = np.cos(2 * np.pi * u1) * np.sqrt(-2 * 0.01 * np.log(u2))
w = 2 / 640; edges = np.linspace(-1, 1, 641); h, _ = np.histogram(g, edges)
c = (edges[:-1] + edges[1:]) / 2
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(c, h, lw=0.6, label="experimental")
ax.plot(c, 8192 * w * np.exp(-c ** 2 / 0.02) / np.sqrt(2 * np.pi * 0.01), "k", lw=1.2, label="ideal")
ax.set_xlim(-1, 1); ax.set_ylim(0, 150); ax.set_xlabel("Range"); ax.set_ylabel("Probability Distribution")
ax.legend(fontsize=7); save(fig, "fig2_3")


# ---- Bayes decision function for a 2-input equaliser (Figs 4.6, 4.7, 5.2, 5.6)
def bayes(ch, X, Y, snr=16):
    var = 10 ** (-snr / 10); f = np.zeros_like(X); pts = []
    for d in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
        cx = ch[0] * d[0] + ch[1] * d[1]; cy = ch[0] * d[1] + ch[1] * d[2]
        f += d[0] * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * var)); pts.append((cx, cy, d[0]))
    return f, pts


def boundary_plot(name, ch, lim=3, noisy=False):
    g = np.linspace(-lim, lim, 400); X, Y = np.meshgrid(g, g); f, pts = bayes(ch, X, Y)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    if noisy:
        r = np.random.default_rng(16); d = r.choice([-1, 1], 1002); s = np.sqrt(10 ** -1.6)
        x = ch[0] * d[2:] + ch[1] * d[1:-1] + s * r.standard_normal(1000)
        xm1 = ch[0] * d[1:-1] + ch[1] * d[:-2] + s * r.standard_normal(1000)
        ax.plot(x, xm1, ".", ms=1.5, color="0.6")
    for cx, cy, s in {(p[0], p[1], p[2]) for p in pts}:
        ax.plot(cx, cy, "bx" if s < 0 else "bo", ms=7)
    ax.contour(X, Y, f, [0], colors="r")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("x(n)"); ax.set_ylabel("x(n-1)"); save(fig, name)


boundary_plot("fig4_6", [1.0, 0.5], noisy=True)
boundary_plot("fig5_2", [1.0, 0.5])
boundary_plot("fig5_6", [0.5, 1.0])
g = np.linspace(-3, 3, 61); X, Y = np.meshgrid(g, g); f, _ = bayes([1.0, 0.5], X, Y)
fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(projection="3d")
ax.plot_wireframe(X, Y, f, lw=0.4, color="k"); ax.set_zlim(-1.5, 1.5)
ax.set_xlabel("x(n)"); ax.set_ylabel("x(n-1)"); ax.set_zlabel("y(n)"); save(fig, "fig4_7")

# ---- Fig 5.3 / 5.4: ADALINE ensemble squared error
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for t in (1, 2, 3, 7, 11):
    y = curve("fig5_3", f"taps={t}")
    if y is not None: ax.semilogy(y, lw=0.7, label=f"{t} tap" + ("s" if t > 1 else ""))
ax.set_ylim(1e-3, 10); ax.set_xlim(0, 1000); ax.set_xlabel("Preamble Bit number"); ax.set_ylabel("Ensemble Square Error")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3); save(fig, "fig5_3")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for e in ("0.1", "0.01", "0.001", "0.0001"):
    y = curve("fig5_4", f"eta={e}", all_trials=True)
    if y is not None: ax.semilogy(y, lw=0.7, label=e)
ax.set_ylim(1e-3, 10); ax.set_xlim(0, 1000); ax.set_xlabel("Preamble Bit Number"); ax.set_ylabel("Ensemble Square Error")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3); save(fig, "fig5_4")

# ---- Fig 5.5 / 5.7 ADALINE BER
for scn in ("fig5_5", "fig5_7"):
    fig, ax = plt.subplots(figsize=(5, 3.4)); plot_ber(ax, *ber(scn), marker="o", color="k", lw=1)
    ber_axes(ax); save(fig, scn)

# ---- Fig 6.1 sigmoid
v = np.linspace(-4, 4, 200); fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(v, np.tanh(v), "k"); ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-1.5, 1.5); ax.set_xlim(-4, 4)
ax.set_xlabel("v(n)"); ax.set_ylabel("sigmoid function - tanh(v(n))"); save(fig, "fig6_1")

# ---- Fig 6.2 backprop adaptation curves (BPBC only runs with the BP-DEBUG fix -> clang_fix)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for ch, ls in (("ch1", "-"), ("ch2", "--")):
    y = curve(f"bp_{ch}", "snr=21")
    if y is not None: ax.semilogy(y, "k" + ls, lw=0.8, label=f"Channel {ch[-1]}")
ax.set_ylim(1e-5, 10); ax.set_xlim(0, 1000); ax.set_xlabel("Epoch number"); ax.set_ylabel("Ensemble Squared Error")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3); save(fig, "fig6_2")

# ---- Fig 7.2 / 7.3 RNN
fig, ax = plt.subplots(figsize=(5.5, 3.6)); y = curve("fig7_2", "snr=21")
if y is not None: ax.semilogy(y, "k", lw=0.7)
ax.set_ylim(lo_decade(y, 0.01), 1.2); ax.set_xlim(0, 1000); ax.set_xlabel("Preamble Bit Number"); ax.set_ylabel("Log10 Ensemble Squared Error")
ax.grid(True, which="both", alpha=0.3); save(fig, "fig7_2")
fig, ax = plt.subplots(figsize=(5, 3.4)); plot_ber(ax, *ber("fig7_3"), marker="o", color="k", lw=1)
ber_axes(ax, ylab="log10 (BER)", lo=1e-5, hi=1, xl=(6, 30)); save(fig, "fig7_3")

# ---- Fig 8.3 RBF
fig, ax = plt.subplots(figsize=(5, 3.4))
plot_ber(ax, *ber("fig8_3_ch1"), marker="D", color="k", lw=1, label="CHANNEL 1")
plot_ber(ax, *ber("fig8_3_ch2"), marker="o", color="k", ls="--", lw=1, label="CHANNEL 2")
ber_axes(ax, ylab="log10 (BER)", lo=1e-5, hi=1, xl=(6, 30)); ax.legend(fontsize=7); save(fig, "fig8_3")

# ---- Fig 9.1 impulse response
fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(range(len(REAL)), REAL, width=0.4, color="k")
ax.set_xlim(-0.5, 20); ax.set_ylim(0, 1); ax.set_xlabel("Time (symbol periods)"); ax.set_ylabel("Received Signal Strength")
save(fig, "fig9_1")

# ---- Fig 9.2 / 9.3 telephone channel. The report's eta=0.08 diverges in the listing (runs hang);
# eta=0.008 (fig9_3_ada008) is plotted. Curves at the highest SNR run (20 dB).
fig, ax = plt.subplots(figsize=(5.5, 3.6)); lo = 0.01
for scn, lab, col in (("fig9_3_rnn", "RNN", "g"), ("fig9_3_ada008", "ADALINE", "r")):
    y = curve(scn, "snr=20")
    if y is not None: ax.semilogy(y, col, lw=0.7, label=lab); lo = lo_decade(y, lo)
ax.set_ylim(lo, 10); ax.set_xlabel("Preamble Bit number"); ax.set_ylabel("Ensemble Squared Error")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3); save(fig, "fig9_2")
fig, ax = plt.subplots(figsize=(5, 3.4))
plot_ber(ax, *ber("fig9_3_ada008"), marker="s", color="r", lw=1, label="ADALINE")
plot_ber(ax, *ber("fig9_3_rnn"), marker="D", color="g", lw=1, label="RNN")
ber_axes(ax, xlab="SNR", lo=1e-6, hi=1, xl=(5, 25)); ax.legend(fontsize=7); save(fig, "fig9_3")

(OUT / "substitutions.txt").write_text("\n".join(subs) + "\n")
print("\n".join(subs) or "no substitutions")
