"""Extra check (not a port): per-sample ("temporal") backprop, which the report says failed
to converge within 1000 bits in >= 1 of 5 trials even without noise. That program is not in
the listing, so this re-creates it from the report's description: 2-6-6-1 tanh MLP with
bias -1 inputs, per-bit gradient step, PN preamble, channel 1, pass test = mean sq. error of
last 400 bits < 0.6 (as in LMS.C/RNN.C), weights 0..0.3 uniform as in BPBC.C.
uv run --no-project --with numpy python scenarios/extra_bp_online.py
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from anneq_numpy import pn_sequence


def trial(rng, h, snr, eta, bits=1000, sizes=(2, 6, 6, 1), preamble="pn", test_bits=10000):
    sig = np.sqrt(10 ** (-snr / 10))
    Ws = [rng.uniform(0, 0.3, (sizes[i + 1], sizes[i] + 1)) for i in range(3)]
    s = pn_sequence(bits) if preamble == "pn" else rng.choice([-1.0, 1.0], bits)
    def run(sym, train):
        full = np.concatenate([-np.ones(len(h) - 1), sym])
        rx = np.convolve(full, h, "valid") + sig * rng.standard_normal(len(sym))
        x = -np.ones(sizes[0]); err = []; wrong = 0
        for n in range(len(sym)):
            x = np.roll(x, 1); x[0] = rx[n]
            a = [np.append(x, -1.0)]
            for W in Ws:
                a.append(np.append(np.tanh(W @ a[-1]), -1.0))
            y = a[-1][0]; e = sym[n] - y
            if train:
                err.append(e * e)
                d = e * (1 - y * y)
                deltas = [np.array([d])]
                for k in (2, 1):
                    back = Ws[k][:, :-1].T @ deltas[0]
                    deltas.insert(0, back * (1 - a[k][:-1] ** 2))
                for k in range(3):
                    Ws[k] += eta * np.outer(deltas[k], a[k])
            else:
                wrong += (np.sign(y) or 1) != sym[n]
        return err, wrong
    err, _ = run(s, True)
    passed = np.mean(err[-400:]) < 0.6
    _, wrong = run(rng.choice([-1.0, 1.0], test_bits), False)
    return passed, np.mean(err[-100:]), wrong / test_bits


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print("snr  eta   fails/20  mean_final_err  mean_BER")
    for snr in (40, 21, 10):
        for eta in (0.01, 0.05, 0.1):
            r = [trial(rng, [1.0, 0.5], snr, eta) for _ in range(20)]
            print(f"{snr:3d} {eta:5.2f}  {sum(not p for p, _, _ in r):3d}       "
                  f"{np.mean([e for _, e, _ in r]):.4f}        {np.mean([b for _, _, b in r]):.5f}")
