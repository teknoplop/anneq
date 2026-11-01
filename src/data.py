"""Training/test symbol sources: i.i.d. random BPSK and maximal-length PN sequences."""
from __future__ import annotations

import numpy as np

# Primitive feedback taps (Fibonacci LFSR, 1-indexed) for m-sequences of degree k.
_TAPS = {5: (5, 3), 7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14)}


def pn_sequence(degree: int, n: int | None = None, seed_state: int = 1) -> np.ndarray:
    """+/-1 m-sequence of period 2^degree - 1 (n symbols, default one period)."""
    taps = _TAPS[degree]
    period = 2**degree - 1
    n = period if n is None else n
    state = seed_state & period or 1
    out = np.empty(n)
    mask = sum(1 << (t - 1) for t in taps)  # Galois form
    for i in range(n):
        b = state & 1
        out[i] = 1.0 if b else -1.0
        state >>= 1
        if b:
            state ^= mask
    return out


def random_symbols(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=n)


def rng_for(*keys: int | str) -> np.random.Generator:
    """Independent, reproducible stream per (experiment, channel, snr, role, trial...)."""
    ints = [k if isinstance(k, int) else int.from_bytes(k.encode()[:8].ljust(8, b"\0"), "little")
            for k in keys]
    return np.random.default_rng(np.random.SeedSequence([2026, *ints]))
