"""Idiomatic numpy version of the 1995 ADALINE/LMS experiment.

All trials run side by side as rows of arrays (time stays a Python loop, since
LMS is sequential). Same experiment as LMS.C: 31-bit PN preamble from the
5-bit register (taps 2 and 5), per-sample LMS w += 2*eta*e*x, decision delay,
retrain while mean squared error over the last 400 bits >= 0.6, then a
10 000-bit BER test on random bits. Noise: numpy Gaussian, var 10^(-SNR/10).
Options not in the original: bias=True feeds the bias input a constant 1
(fixing the 1995 bug), preamble="random" uses random training bits.
"""
import numpy as np


def pn_sequence(n):
    reg = [1, 1, 1, 1, 1]
    out = np.empty(n)
    out[0] = 1.0                       # LMS.C sends a start bit of +1 first
    for k in range(1, n):
        s = reg[1] ^ reg[4]
        reg = [s] + reg[:4]
        out[k] = 1.0 if s else -1.0
    return out


def run_lms(channel, taps, eta, snr, delay=0, bits=1000, trials=100,
            test_bits=10000, bias=False, preamble="pn", seed=0, retrain=True, max_rounds=30):
    """max_rounds caps the retrain loop (LMS.C has no cap and can loop forever);
    trials still failing after it are kept as they are."""
    rng = np.random.default_rng(seed)
    h = np.asarray(channel, float)
    sigma = np.sqrt(10.0 ** (-snr / 10.0))
    T = trials
    curves = np.zeros((T, bits - delay))
    W = np.zeros((T, taps + 1))
    pending = np.ones(T, bool)
    fails = 0
    rounds = 0
    while pending.any():
        rounds += 1
        idx = np.flatnonzero(pending)
        n = idx.size
        if preamble == "pn":
            sym = np.tile(pn_sequence(bits), (n, 1))
        else:
            sym = rng.choice([-1.0, 1.0], size=(n, bits))
        sym = np.hstack([-np.ones((n, len(h) - 1)), sym])   # channel line starts at -1
        rx = np.stack([np.convolve(s, h, "valid") for s in sym]) + sigma * rng.standard_normal((n, bits))
        x = -np.ones((n, taps + 1))
        x[:, taps] = 1.0 if bias else 0.0
        w = rng.uniform(0, 0.1, (n, taps + 1))
        des = sym[:, len(h) - 1:]
        cur = np.zeros((n, bits - delay))
        for b in range(bits):
            x[:, 1:taps] = x[:, 0:taps - 1].copy()
            x[:, 0] = rx[:, b]
            if b >= delay:
                e = des[:, b - delay] - (x * w).sum(1)
                cur[:, b - delay] = e * e
                w += 2.0 * eta * e[:, None] * x
        ok = cur[:, -400:].mean(1) < 0.6 if retrain else np.ones(n, bool)
        ok = np.where(np.isfinite(cur[:, -400:].mean(1)), ok, False)
        if rounds >= max_rounds:
            fails += int((~ok).sum()); ok[:] = True
        fails += int((~ok).sum())
        curves[idx[ok]] = cur[ok]
        W[idx[ok]] = w[ok]
        pending[idx[ok]] = False
    # BER test: fresh random bits; the channel/equaliser lines carry on from training
    # in LMS.C, which only matters for the first few of 10 000 bits.
    s = rng.choice([-1.0, 1.0], size=(T, test_bits + len(h) - 1 + taps))
    rx = np.stack([np.convolve(r, h, "valid") for r in s]) + sigma * rng.standard_normal((T, test_bits + taps))
    X = np.lib.stride_tricks.sliding_window_view(rx, taps, axis=1)[:, :, ::-1][:, :test_bits]
    y = (X * W[:, None, :taps]).sum(2) + W[:, None, taps] * (1.0 if bias else 0.0)
    des = s[:, len(h) - 1 + taps - 1 - delay: len(h) - 1 + taps - 1 - delay + test_bits]
    dec = np.where(y < 0, -1.0, 1.0)
    ber = (dec != des).mean(1)
    return curves.mean(0), ber.mean(), fails / T
