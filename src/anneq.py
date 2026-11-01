#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.5", "scipy>=1.18"]
# ///
"""The four 1995 equaliser simulators, written as idiomatic modern Python.

    uv run src/anneq.py {lms,rnn,rbf,bp} --channel ch1 --snr 10 [--curve out.csv] ...

Same problem, networks and training rules as the 1995 C (LMS.C, RNN.C, BPBC.C, RB.C; the
behavioural reference is src/anneq_fixed.py, the line port with issues #10-#18 fixed), written
as if Python 3.15 and today's numpy/scipy had existed in 1995. No modern training methods: no
Adam, no autograd, no BPTT, no gated cells. Section and equation numbers are those of the report
(docs/report_text.md).

Experiment (§3.2): per trial, a 1000-bit training preamble, then a 10 000-bit BER test on random
bits, repeated 100 times. LMS and RNN trials whose mean squared error over the last 400 preamble
bits is >= 0.6 are reset and retrained, at most 10 attempts (#14); a trial that never passes still
counts in the BER, but not in the learning curve. The preamble is random +-1 bits by default (#16);
--preamble pn gives the report's 31-bit PN sequence (§3.3).

Library or own code, per component:
  channel, ISI (§2.3)    numpy.convolve over the transmitted bits
  noise (§2.2, eq 2-4)   numpy.random.Generator.normal (replaces the Box-Muller on rand())
  PN preamble (§3.3)     scipy.signal.max_len_seq(5): the same 31-bit sequence as the 1995 register
  tapped delay lines     numpy.lib.stride_tricks.sliding_window_view
  ADALINE / LMS (§5)     own loop. padasip's FilterLMS is the same rule, but it is unmaintained
                         (last release 2022, SyntaxWarnings on 3.15) and saves only five lines
  backprop MLP (§6.3)    own vectorised numpy. sklearn's MLPRegressor has a linear output node,
                         averages the gradient, adds L2 and uses v = mc*v - lr*g, so it is not
                         BPBC's tanh-output, summed-gradient, (1-mc)-scaled momentum rule
  RNN / RTRL (§7)        own code: no maintained library implements RTRL
  RBF centres (§8)       own code (per-state running mean via numpy.add.at); distances in the
                         test by scipy.spatial.distance.cdist
"""
from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import max_len_seq
from scipy.spatial.distance import cdist

type Array = NDArray[np.float64]

CHANNELS: dict[str, tuple[float, ...]] = {
    "ch1": (1.0, 0.5),                                       # §3.4, channel 1
    "ch2": (0.5, 1.0),                                       # §3.4, channel 2 (non-minimum phase)
    "tel": (0.047, 0.321, 0.783, 0.427, 0.293, 0.100,        # §9, Fig 9.1: British telephone line
            0.041, 0.038, 0.020, 0.011),
}
PASS_MSE = 0.6        # §3.2 training check: mean squared error over the last 400 preamble bits
PASS_WINDOW = 400
FINAL_WINDOW = 100    # "final error" = mean of the last 100 points of the learning curve


# --- configuration --------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Experiment:
    """What every simulator shares: channel, noise, preamble, test length and trials (§3.2)."""
    channel: str = "ch1"
    snr_db: float = 10.0
    delay: int = 0                 # decision delay: the equaliser estimates d(n - delay)
    train_bits: int = 1000
    test_bits: int = 10_000
    trials: int = 100
    seed: int = 1000
    preamble: str = "random"       # "random" (#16) or "pn" (§3.3)
    max_attempts: int = 10         # #14: cap on retraining a failed LMS/RNN trial

    @property
    def taps(self) -> Array:
        return np.asarray(CHANNELS[self.channel])


@dataclass(frozen=True)
class LMS:
    """ADALINE (Fig 5.1): `taps` inputs plus a -1 bias, trained by LMS with rate `eta`."""
    taps: int = 7
    eta: float = 0.01


@dataclass(frozen=True)
class RNN:
    """Fully recurrent network (Fig 7.1): `inputs` taps, bias, `nodes` fed-back outputs."""
    inputs: int = 2
    nodes: int = 3
    eta: float = 0.12              # re-tuned for the corrected RTRL (#11); 1995 used 0.005


@dataclass(frozen=True)
class RBF:
    """RBF equaliser (Fig 8.2): one Gaussian node per channel state, fixed +-1 output weights."""
    inputs: int = 2
    centres: int = 8
    width: float = 0.125           # 2*sigma_w^2 in eq 8.2


@dataclass(frozen=True)
class BP:
    """BPBC (§6.3): MLP `layers` (inputs, hidden...), one tanh output, batch backprop."""
    layers: tuple[int, ...] = (2, 6, 6)
    lr: float = 0.1
    momentum: float = 0.5
    epochs: int = 1000
    patterns: int = 40


@dataclass
class Trial:
    ber: float
    curve: Array | None            # squared error per preamble bit (or per epoch for BP)
    passed: bool = True
    attempts: int = 1


@dataclass
class Summary:
    trials: list[Trial] = field(default_factory=list)

    @property
    def ber(self) -> float:
        return float(np.mean([t.ber for t in self.trials]))

    @property
    def curve(self) -> Array | None:
        """Ensemble squared error (§3.2): the mean curve over the trials that trained."""
        curves = [t.curve for t in self.trials if t.passed and t.curve is not None]
        return np.mean(curves, axis=0) if curves else None

    @property
    def final_err(self) -> float:
        c = self.curve
        return float(np.mean(c[-FINAL_WINDOW:])) if c is not None else float("nan")

    @property
    def fails_per_trial(self) -> float:
        return sum(t.attempts - t.passed for t in self.trials) / len(self.trials)

    @property
    def gaveup(self) -> float:
        return sum(not t.passed for t in self.trials) / len(self.trials)


# --- signals (§2, §3) -----------------------------------------------------------------------
def noise_sigma(snr_db: float) -> float:
    """§2.2: SNR = 10 log10(1/sigma^2) for +-1 symbols, so sigma = 10^(-SNR/20)."""
    return 10.0 ** (-snr_db / 20.0)


def pn_preamble(n: int) -> Array:
    """§3.3: a +1 start bit, then the 5-bit register's 31-bit PN sequence (taps 2 and 5) as +-1.

    max_len_seq(5) gives the same m-sequence; the 1995 register's output starts 5 bits in.
    """
    seq = max_len_seq(5, length=n + 4)[0][5:] * 2.0 - 1.0
    return np.concatenate(([1.0], seq))


def random_bits(rng: np.random.Generator, n: int) -> Array:
    return rng.choice((-1.0, 1.0), size=n)


def transmit(exp: Experiment, rng: np.random.Generator) -> tuple[Array, Array]:
    """One trial's symbols d(n), preamble then test bits, and the received x(n) (eq 2-3).

    The channel starts with -1 in its delay line, as in 1995.
    """
    pre = pn_preamble(exp.train_bits) if exp.preamble == "pn" else random_bits(rng, exp.train_bits)
    d = np.concatenate((pre, random_bits(rng, exp.test_bits)))
    h = exp.taps
    isi = np.convolve(np.concatenate((np.full(len(h) - 1, -1.0), d)), h, mode="valid")
    return d, isi + rng.normal(0.0, noise_sigma(exp.snr_db), d.size)


def tapped_delay(x: Array, n: int, fill: float = -1.0) -> Array:
    """Row t is the delay line [x(t), x(t-1), ..., x(t-n+1)] (Fig 4.5), started full of `fill`."""
    padded = np.concatenate((np.full(n - 1, fill), x))
    return np.lib.stride_tricks.sliding_window_view(padded, n)[:, ::-1]


def with_bias(X: Array) -> Array:
    """Append the -1 bias input (Fig 5.1, Fig 7.1)."""
    return np.concatenate((X, np.full((*X.shape[:-1], 1), -1.0)), axis=-1)


def decide(y: Array) -> Array:
    """Hard decision; 0 and NaN (a diverged network) count as +1, as C's sign() did."""
    return np.where(y < 0.0, -1.0, 1.0)


def ber(y: Array, d: Array) -> float:
    return float(np.mean(decide(y) != d))


def trained(sq_err: Array) -> bool:
    """§3.2: the trial trained if the mean squared error over the last 400 bits is < 0.6."""
    return bool(np.mean(sq_err[-PASS_WINDOW:]) < PASS_MSE)


# --- ADALINE / LMS (§5) ---------------------------------------------------------------------
def lms_train(X: Array, d: Array, w: Array, eta: float) -> tuple[Array, Array]:
    """§5.2: e(n) = d(n) - y(n), w(n+1) = w(n) + 2 eta e(n) x(n). Inherently sequential."""
    w, sq_err = w.copy(), np.empty(len(d))
    for n, (x, dn) in enumerate(zip(X, d)):
        e = dn - x @ w
        w += 2.0 * eta * e * x
        sq_err[n] = e * e
    return w, sq_err


def lms_trial(cfg: LMS, exp: Experiment, rng: np.random.Generator) -> Trial:
    for attempt in range(1, exp.max_attempts + 1):
        d, x = transmit(exp, rng)
        X = with_bias(tapped_delay(x, cfg.taps))
        n, k = exp.train_bits, exp.delay
        w0 = rng.uniform(0.0, 0.1, cfg.taps + 1)              # 1995 fill_rand: U(0, 0.1)
        w, sq_err = lms_train(X[k:n], d[:n - k], w0, cfg.eta)
        if ok := trained(sq_err):
            break
    return Trial(ber(X[n:] @ w, d[n - k:-k or None]), sq_err, ok, attempt)


# --- recurrent network, RTRL (§7) -----------------------------------------------------------
@dataclass
class RecurrentState:
    """Weights W[k, l] from input l to node k, outputs y, sensitivities p[k, l, j] = dy_j/dw_kl."""
    W: Array
    y: Array
    p: Array

    @classmethod
    def initial(cls, cfg: RNN, rng: np.random.Generator) -> RecurrentState:
        n_in = cfg.inputs + 1 + cfg.nodes
        return cls(rng.uniform(0.0, 0.1, (cfg.nodes, n_in)), np.zeros(cfg.nodes),
                   np.zeros((cfg.nodes, n_in, cfg.nodes)))      # pi(0) = 0 (§7.2)


def rnn_inputs(x: Array, y: Array) -> Array:
    """Fig 7.1 input layer: the taps, the -1 bias and last step's outputs fed back (#10)."""
    return np.concatenate((x, (-1.0,), y))


def rtrl_step(s: RecurrentState, x: Array, d: float, eta: float) -> float:
    """One RTRL step (§7.2, steps 1-4); returns e(n)^2. Updates `s` in place.

    Eq 7.1: p_j^{kl}(n) = phi'(v_j) [ sum_i w_{j,i} p_i^{kl}(n-1) + delta_kj u_l(n) ], the sum
    over the fed-back outputs i, the delta*u term counted once (#11); phi = tanh, phi' = 1 - y^2
    (eqs 6-4, 6-5). Then w_kl += eta e(n) p_0^{kl}.
    """
    u = rnn_inputs(x, s.y)
    y = np.tanh(s.W @ u)
    e = d - y[0]
    fb = s.W[:, -len(y):]                               # w_{j,i}: weights on the fed-back inputs
    total = s.p @ fb.T                                  # [k, l, j] = sum_i p[k, l, i] w[j, i]
    diag = np.arange(len(y))
    total[diag, :, diag] += u                           # delta_kj u_l
    s.p = (1.0 - y * y) * total
    s.W += eta * e * s.p[:, :, 0]
    s.y = y
    return e * e


def rnn_run(s: RecurrentState, x: Array) -> float:
    """Feed one input vector forward without learning; returns y_0(n)."""
    s.y = np.tanh(s.W @ rnn_inputs(x, s.y))
    return s.y[0]


def rnn_trial(cfg: RNN, exp: Experiment, rng: np.random.Generator) -> Trial:
    n, k = exp.train_bits, exp.delay
    for attempt in range(1, exp.max_attempts + 1):
        d, x = transmit(exp, rng)
        X = tapped_delay(x, cfg.inputs)
        s = RecurrentState.initial(cfg, rng)
        sq_err = np.array([rtrl_step(s, X[t], d[t - k], cfg.eta) for t in range(k, n)])
        if ok := trained(sq_err):
            break
    y = np.array([rnn_run(s, xt) for xt in X[n:]])      # sequential: the outputs feed back
    return Trial(ber(y, d[n - k:-k or None]), sq_err, ok, attempt)


# --- RBF equaliser (§8) ---------------------------------------------------------------------
def channel_states(d: Array, memory: int) -> Array:
    """Index of the RBF node that should fire: the last `memory` symbols as a binary number,
    newest first (+1 -> 1), so the upper half of the nodes are the d(n) = +1 states (§8.2)."""
    bits = tapped_delay(d, memory) > 0
    return bits @ (1 << np.arange(memory - 1, -1, -1))


def rbf_centres(X: Array, states: Array, n_centres: int) -> Array:
    """§8.2 training: each node's centre is the mean of the inputs seen in its state.
    Nodes never visited keep a centre at the origin, as in 1995."""
    sums = np.zeros((n_centres, X.shape[1]))
    np.add.at(sums, states, X)
    counts = np.bincount(states, minlength=n_centres)[:, None]
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def rbf_output(X: Array, centres: Array, width: float) -> Array:
    """Eqs 8.1/8.2 (as coded: exp(-d^2/width)), summed by the ADALINE with weights -1 / +1."""
    w = np.where(np.arange(len(centres)) < len(centres) // 2, -1.0, 1.0)
    return np.exp(-cdist(X, centres, "sqeuclidean") / width) @ w


def rbf_trial(cfg: RBF, exp: Experiment, rng: np.random.Generator) -> Trial:
    d, x = transmit(exp, rng)
    X, n = tapped_delay(x, cfg.inputs), exp.train_bits
    states = channel_states(d, int(np.log2(cfg.centres)))
    train = slice(cfg.inputs, n)              # 1995 waits until the delay line is full
    centres = rbf_centres(X[train], states[train], cfg.centres)
    return Trial(ber(rbf_output(X[n:], centres, cfg.width), d[n:]), None)


# --- backprop MLP, batch mode with momentum (§6) --------------------------------------------
def mlp_forward(Ws: list[Array], X: Array) -> list[Array]:
    """Activations of every layer for all patterns at once (eq 6-4, tanh, -1 bias inputs)."""
    acts = [X]
    for W in Ws:
        acts.append(np.tanh(with_bias(acts[-1]) @ W.T))
    return acts


def mlp_gradients(Ws: list[Array], acts: list[Array], e: Array) -> list[Array]:
    """Local gradients (eqs 6-2, 6-3) summed over the epoch, as BPBC sums delta*y."""
    delta = e[:, None] * (1.0 - acts[-1] ** 2)
    grads = []
    for W, a in zip(reversed(Ws), reversed(acts[:-1])):
        grads.append(delta.T @ with_bias(a))
        delta = (delta @ W[:, :-1]) * (1.0 - a * a)
    return grads[::-1]


def bp_train(cfg: BP, X: Array, d: Array, rng: np.random.Generator) -> tuple[list[Array], Array]:
    """§6.3: one weight update per epoch; dw = lr (1 - mc) g + mc dw_prev (eq 6-1 + momentum)."""
    sizes = (*cfg.layers, 1)
    Ws = [rng.uniform(0.0, 0.3, (m, n + 1)) for n, m in zip(sizes, sizes[1:])]
    steps = [np.zeros_like(W) for W in Ws]
    mse = np.empty(cfg.epochs)
    for epoch in range(cfg.epochs):
        acts = mlp_forward(Ws, X)
        e = d - acts[-1][:, 0]
        mse[epoch] = np.mean(e * e)
        for W, step, g in zip(Ws, steps, mlp_gradients(Ws, acts, e)):
            step *= cfg.momentum
            step += cfg.lr * (1.0 - cfg.momentum) * g
            W += step
    return Ws, mse


def bp_trial(cfg: BP, exp: Experiment, rng: np.random.Generator) -> Trial:
    first = -(-cfg.layers[0] // 2)                      # BPBC skips ceil(inputs/2) samples
    exp = replace(exp, train_bits=first + cfg.patterns)
    d, x = transmit(exp, rng)
    X, n = tapped_delay(x, cfg.layers[0]), exp.train_bits
    Ws, mse = bp_train(cfg, X[first:n], d[first:n], rng)
    return Trial(ber(mlp_forward(Ws, X[n:])[-1][:, 0], d[n:]), mse)


# --- experiment driver and CLI --------------------------------------------------------------
type TrialFn = Callable[[np.random.Generator], Trial]


def run(trial: TrialFn, exp: Experiment) -> Summary:
    """§3.2: `exp.trials` independent trials, each with its own seeded generator."""
    seeds = np.random.SeedSequence(exp.seed).spawn(exp.trials)
    with np.errstate(over="ignore", invalid="ignore"):   # a diverging LMS (eta too big) is a result
        return Summary([trial(np.random.default_rng(s)) for s in seeds])


def simulate(model: LMS | RNN | RBF | BP, exp: Experiment) -> Summary:
    trial_fn = {LMS: lms_trial, RNN: rnn_trial, RBF: rbf_trial, BP: bp_trial}[type(model)]
    return run(lambda rng: trial_fn(model, exp, rng), exp)


def write_curve(path: str, curve: Array) -> None:
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("n", "squared_error"))
        out.writerows((i, f"{v:.7g}") for i, v in enumerate(curve))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--channel", choices=CHANNELS, default="ch1")
    common.add_argument("--snr", type=float, default=10.0, help="dB")
    common.add_argument("--delay", type=int, default=0, help="decision delay (lms, rnn)")
    common.add_argument("--trials", type=int, default=100)
    common.add_argument("--seed", type=int, default=1000)
    common.add_argument("--train-bits", type=int, default=1000)
    common.add_argument("--test-bits", type=int, default=10_000)
    common.add_argument("--preamble", choices=("random", "pn"), default="random")
    common.add_argument("--curve", metavar="CSV", help="write the learning curve here")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="model", required=True)
    p = sub.add_parser("lms", parents=[common], help="ADALINE trained by LMS (§5)")
    p.add_argument("--taps", type=int, default=7)
    p.add_argument("--eta", type=float, default=0.01)
    p = sub.add_parser("rnn", parents=[common], help="recurrent network trained by RTRL (§7)")
    p.add_argument("--inputs", type=int, default=2)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.12)
    p = sub.add_parser("rbf", parents=[common], help="RBF equaliser (§8)")
    p.add_argument("--inputs", type=int, default=2)
    p.add_argument("--centres", type=int, default=8)
    p.add_argument("--width", type=float, default=0.125)
    p = sub.add_parser("bp", parents=[common], help="batch backprop MLP, BPBC (§6.3)")
    p.add_argument("--layers", type=int, nargs="+", default=[2, 6, 6])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patterns", type=int, default=40)
    return ap.parse_args(argv)


def model_from(a: argparse.Namespace) -> LMS | RNN | RBF | BP:
    match a.model:
        case "lms": return LMS(a.taps, a.eta)
        case "rnn": return RNN(a.inputs, a.nodes, a.eta)
        case "rbf": return RBF(a.inputs, a.centres, a.width)
        case _: return BP(tuple(a.layers), a.lr, a.momentum, a.epochs, a.patterns)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    exp = Experiment(channel=a.channel, snr_db=a.snr, delay=a.delay, train_bits=a.train_bits,
                     test_bits=a.test_bits, trials=a.trials, seed=a.seed, preamble=a.preamble)
    model = model_from(a)
    s = simulate(model, exp)
    print(f"{model}  {a.channel} {a.snr:g} dB, {exp.trials} trials")
    print(f"BER {s.ber:.4g}  final error {s.final_err:.4g}  "
          f"failed attempts/trial {s.fails_per_trial:.2f}  gave up {s.gaveup:.2f}")
    if a.curve and (c := s.curve) is not None:
        write_curve(a.curve, c)


if __name__ == "__main__":
    main()
