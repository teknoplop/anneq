"""Fast checks of src/anneq.py (a few seconds in all).

uv run --python 3.15 --no-project --with numpy --with scipy --with pytest pytest -q src/tests
"""
import pathlib, sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import anneq  # noqa: E402


def test_pn_preamble_is_the_1995_register():
    reg, out = [1, 1, 1, 1, 1], [1.0]                # LMS.C: start bit +1, then taps 2 xor 5
    for _ in range(61):
        s = reg[1] ^ reg[4]
        reg = [s] + reg[:4]
        out.append(1.0 if s else -1.0)
    assert np.array_equal(anneq.pn_preamble(62), out)


def test_noise_variance_matches_snr():
    rng = np.random.default_rng(0)
    exp = anneq.Experiment(channel="ch1", snr_db=10, train_bits=0, test_bits=200_000)
    d, x = anneq.transmit(exp, rng)
    isi = x - np.convolve(np.r_[-1.0, d], exp.taps, "valid")
    assert np.var(isi) == pytest.approx(0.1, rel=0.02)


def test_tapped_delay_rows_newest_first():
    X = anneq.tapped_delay(np.array([1.0, 2.0, 3.0]), 3)
    assert X.tolist() == [[1, -1, -1], [2, 1, -1], [3, 2, 1]]


def test_rtrl_sensitivities_are_the_exact_gradient():
    """With eta = 0, p[k, l, 0] after T steps must equal dy_0(T)/dW[k, l] (eq 7.1, #10, #11)."""
    cfg, rng = anneq.RNN(inputs=2, nodes=3), np.random.default_rng(1)
    X = rng.normal(size=(12, cfg.inputs))
    s0 = anneq.RecurrentState.initial(cfg, rng)
    s0.W = rng.normal(0, 0.5, s0.W.shape)

    def final_y0(W):
        s = anneq.RecurrentState(W.copy(), np.zeros(cfg.nodes), np.zeros_like(s0.p))
        for x in X:
            anneq.rtrl_step(s, x, 0.0, eta=0.0)
        return s

    s, h = final_y0(s0.W), 1e-6
    num = np.zeros_like(s0.W)
    for idx in np.ndindex(*s0.W.shape):
        Wp, Wm = s0.W.copy(), s0.W.copy()
        Wp[idx] += h; Wm[idx] -= h
        num[idx] = (final_y0(Wp).y[0] - final_y0(Wm).y[0]) / (2 * h)
    assert np.allclose(s.p[:, :, 0], num, atol=1e-7)


def test_backprop_gradient_is_minus_dE_dW():
    rng = np.random.default_rng(2)
    X, d = rng.normal(size=(40, 2)), rng.choice((-1.0, 1.0), 40)
    Ws = [rng.normal(0, 0.5, s) for s in ((6, 3), (6, 7), (1, 7))]

    def E(Ws):
        return 0.5 * np.sum((d - anneq.mlp_forward(Ws, X)[-1][:, 0]) ** 2)

    acts = anneq.mlp_forward(Ws, X)
    grads = anneq.mlp_gradients(Ws, acts, d - acts[-1][:, 0])
    h = 1e-6
    for W, g in zip(Ws, grads):
        idx = (1, 2) if W.shape[0] > 1 else (0, 3)
        W[idx] += h; ep = E(Ws); W[idx] -= 2 * h; em = E(Ws); W[idx] += h
        assert g[idx] == pytest.approx(-(ep - em) / (2 * h), rel=1e-5)


def test_adaline_ber_ch1_10db():
    """Fig 5.5: 7-tap ADALINE on channel 1 at 10 dB; fixed line port 3.60e-3 over 100 trials."""
    s = anneq.simulate(anneq.LMS(taps=7, eta=0.01), anneq.Experiment(snr_db=10, trials=30))
    assert 2.8e-3 < s.ber < 4.5e-3
    assert s.final_err == pytest.approx(0.132, rel=0.1)


def test_rbf_ber_ch1_10db():
    """Fig 8.3: RBF on channel 1 at 10 dB; fixed line port 3.71e-3."""
    s = anneq.simulate(anneq.RBF(), anneq.Experiment(snr_db=10, trials=30))
    assert 2.9e-3 < s.ber < 4.6e-3


def test_rnn_trains_on_ch1():
    """Fig 7.3: RNN on channel 1 at 10 dB; fixed line port 1.5e-3 (short test to stay fast)."""
    s = anneq.simulate(anneq.RNN(), anneq.Experiment(snr_db=10, trials=5, test_bits=2000))
    assert s.gaveup == 0 and s.ber < 4e-3
