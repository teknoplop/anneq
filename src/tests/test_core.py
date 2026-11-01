import numpy as np
import pytest

import linear, trellis
from channels import CHANNELS, mmse_delay, transmit, windows
from data import pn_sequence, random_symbols, rng_for
from metrics import clopper_pearson, count_errors, wilson


def test_channel_convolution_matches_definition():
    ch = CHANNELS["ch1"]
    rng = np.random.default_rng(0)
    s = random_symbols(rng, 1000)
    full = np.concatenate([[1.0], s])
    x = ch.noiseless(full)
    assert np.allclose(x[1:], s[1:] + 0.5 * s[:-1])
    assert np.isclose(x[0], s[0] + 0.5)


@pytest.mark.parametrize("name", list(CHANNELS))
def test_noise_variance_matches_snr(name):
    ch = CHANNELS[name]
    rng = np.random.default_rng(1)
    s, x = transmit(ch, 400_000, 10.0, rng)
    clean = ch.noiseless(np.concatenate([np.ones(ch.L - 1), s]))
    noise = x - clean
    n_lead = ch.L - 1  # first samples used a different history
    ratio = np.mean(clean[n_lead:] ** 2) / np.var(noise)
    assert 10 * np.log10(ratio) == pytest.approx(10.0, abs=0.05)


def test_signal_power_linear_is_tap_energy():
    ch = CHANNELS["ch3"]
    assert ch.signal_power() == pytest.approx(np.sum(np.square(ch.h)))


@pytest.mark.parametrize("k", [5, 7, 9, 11, 15])
def test_pn_is_m_sequence(k):
    P = 2**k - 1
    p = pn_sequence(k)
    assert p.sum() == 1
    ac = {p @ np.roll(p, s) for s in range(1, P)}
    assert ac == {-1}


def test_random_symbols_balanced_and_white():
    s = random_symbols(rng_for("t", 1), 200_000)
    assert abs(s.mean()) < 0.01
    assert abs(np.mean(s[1:] * s[:-1])) < 0.01


def test_rng_streams_are_independent_and_reproducible():
    a = rng_for("x", 1).standard_normal(5)
    assert np.array_equal(a, rng_for("x", 1).standard_normal(5))
    assert not np.array_equal(a, rng_for("x", 2).standard_normal(5))


def test_ber_counter_and_cis():
    assert count_errors(np.array([1, -1, 0.3, -2]), np.array([1, 1, -1, -1])) == 2
    lo, hi = clopper_pearson(100, 100_000)
    assert lo < 1e-3 < hi and hi - lo < 5e-4
    wl, wh = wilson(0, 1000)
    assert wl == pytest.approx(0, abs=1e-12) and 0 < wh < 0.005
    assert clopper_pearson(0, 1000)[1] == pytest.approx(1 - 0.025 ** (1 / 1000), rel=1e-6)


def test_lms_converges_to_wiener_solution():
    ch = CHANNELS["ch1"]
    m, d, snr = 7, 0, 20.0
    s, x = transmit(ch, 60_000, snr, np.random.default_rng(3))
    X, y, _ = windows(x, s, m, d)
    A = np.hstack([X, np.ones((len(X), 1))])
    w_opt = np.linalg.solve(A.T @ A, A.T @ y)
    eq = linear.LMS(m, mu=0.01, passes=3).fit(X, y)
    assert np.linalg.norm(eq.w - w_opt) < 0.05
    # known case: inverse of 1 + 0.5 z^-1 is 1, -0.5, 0.25, ...
    assert eq.w[:3] == pytest.approx([1, -0.5, 0.25], abs=0.06)
    rls = linear.RLS(m).fit(X[:2000], y[:2000])
    assert np.linalg.norm(rls.w - w_opt) < 0.1


def test_mmse_delay_nonminimum_phase_needs_delay():
    assert mmse_delay(CHANNELS["ch1"], 7, 12) <= 2
    assert mmse_delay(CHANNELS["ch2"], 7, 12) >= 4


def test_mlse_noiseless_is_error_free_and_beats_threshold():
    for name in ["ch2", "ch3", "ch1nl"]:
        ch = CHANNELS[name]
        s, x = transmit(ch, 5000, 200.0, np.random.default_rng(4))
        assert count_errors(trellis.mlse(x, ch)[50:], s[50:]) == 0


def test_map_window_beats_linear():
    ch = CHANNELS["ch2"]
    m, d, snr = 7, 5, 8.0
    s, x = transmit(ch, 100_000, snr, np.random.default_rng(5))
    X, y, _ = windows(x, s, m, d)
    z = trellis.MAPWindow(ch, m, d, ch.sigma(snr)).predict(X)
    lin = linear.RLS(m).fit(X[:5000], y[:5000]).predict(X)
    assert count_errors(z, y) < count_errors(lin, y)


def test_rls_stays_stable_on_long_runs():
    ch = CHANNELS["ch3"]
    s, x = transmit(ch, 100_000, 16.0, np.random.default_rng(6))
    X, y, _ = windows(x, s, 15, 8)
    eq = linear.RLS(15).fit(X, y)
    assert count_errors(eq.predict(X[-20000:]), y[-20000:]) / 20000 < 0.01
