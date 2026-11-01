"""Channel models: FIR ISI channel, optional memoryless nonlinearity, AWGN.

SNR is defined at the receiver input as  SNR = E[|f(h * s)|^2] / sigma^2  for BPSK
symbols s in {-1, +1}. The signal power is computed exactly by enumerating every
channel state, so it is correct for the nonlinear channel too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TELEPHONE = [0.047, 0.321, 0.783, 0.427, 0.293, 0.100, 0.041, 0.038, 0.020, 0.011]


@dataclass(frozen=True)
class Channel:
    name: str
    h: tuple[float, ...]
    nl: str | None = None  # None or "poly" (x + 0.2x^2 - 0.1x^3)
    desc: str = field(default="", compare=False)

    @property
    def L(self) -> int:
        return len(self.h)

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.nl is None:
            return x
        if self.nl == "poly":
            return x + 0.2 * x**2 - 0.1 * x**3
        raise ValueError(self.nl)

    def noiseless(self, s: np.ndarray) -> np.ndarray:
        """s has L-1 extra leading symbols (channel history); output has len(s)-L+1."""
        lin = np.convolve(s, np.asarray(self.h), mode="valid")
        return self.f(lin)

    def all_outputs(self) -> np.ndarray:
        """Noiseless output for every state index k (bit j of k = (s[n-j] == +1))."""
        return state_outputs(self)

    def signal_power(self) -> float:
        return float(np.mean(self.all_outputs() ** 2))

    def sigma(self, snr_db: float) -> float:
        return float(np.sqrt(self.signal_power() / 10 ** (snr_db / 10)))


def state_outputs(ch: Channel) -> np.ndarray:
    L = ch.L
    k = np.arange(2**L)
    bits = (k[:, None] >> np.arange(L)[None, :]) & 1  # bit j -> s[n-j]
    s = 2.0 * bits - 1.0
    return ch.f(s @ np.asarray(ch.h))


CHANNELS: dict[str, Channel] = {
    "ch1": Channel("ch1", (1.0, 0.5), desc="H = 1 + 0.5 z^-1 (minimum phase)"),
    "ch2": Channel("ch2", (0.5, 1.0), desc="H = 0.5 + z^-1 (non-minimum phase)"),
    "ch3": Channel("ch3", tuple(TELEPHONE), desc="10-tap telephone channel"),
    "ch1nl": Channel("ch1nl", (1.0, 0.5), "poly", desc="ch1 then y = x + 0.2x^2 - 0.1x^3"),
}


def transmit(ch: Channel, n: int, snr_db: float, rng: np.random.Generator,
             symbols: np.ndarray | None = None):
    """Return (s, x): n symbols and n received samples, x[i] depends on s[i], s[i-1], ...

    History symbols before s[0] are random and not returned.
    """
    if symbols is None:
        symbols = rng.choice([-1.0, 1.0], size=n)
    hist = rng.choice([-1.0, 1.0], size=ch.L - 1)
    full = np.concatenate([hist, symbols])
    x = ch.noiseless(full) + ch.sigma(snr_db) * rng.standard_normal(n)
    return symbols.astype(np.float64), x


def windows(x: np.ndarray, s: np.ndarray, m: int, d: int):
    """Tapped-delay-line windows X[i] = [x[n], x[n-1], ..., x[n-m+1]] with target s[n-d].

    Rows start at n = max(m-1, d) so every row is fully defined.
    """
    n0 = max(m - 1, d)
    idx = np.arange(n0, len(x))
    X = np.stack([x[idx - j] for j in range(m)], axis=1)
    return X, s[idx - d], idx


def mmse_delay(ch: Channel, m: int, snr_db: float) -> int:
    """Decision delay that minimises the MMSE of a length-m linear equaliser (linearised channel)."""
    h = np.asarray(ch.h)
    L = len(h)
    H = np.zeros((m, m + L - 1))
    for i in range(m):
        H[i, i:i + L] = h
    s2 = ch.sigma(snr_db) ** 2
    R = H @ H.T + s2 * np.eye(m)
    best, bd = np.inf, 0
    for d in range(m + L - 1):
        p = H[:, d]
        mse = 1 - p @ np.linalg.solve(R, p)
        if mse < best - 1e-12:
            best, bd = mse, d
    return bd


__all__ = ["Channel", "CHANNELS", "transmit", "windows", "mmse_delay", "state_outputs"]
