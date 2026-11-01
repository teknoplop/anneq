"""Optimum references with known channel (including the nonlinearity):

* MLSE: Viterbi sequence detection, Euclidean branch metric, block decoding with overlap.
* MAP symbol-by-symbol (Bayesian) decision on an m-sample window with delay d. This is the
  best any equaliser with that window/delay can do, i.e. the bound for the MLP/CNN/transformer
  and the structure the RBF equaliser approximates (Chen, Mulgrew & Grant, 1993).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from channels import Channel, state_outputs


@njit(cache=True)
def _viterbi_block(x, out, L):
    S = 1 << (L - 1)
    T = x.shape[0]
    metric = np.zeros(S)
    surv = np.empty((T, S), dtype=np.int32)
    new = np.empty(S)
    for t in range(T):
        for ns in range(S):
            best = 1e300
            bo = 0
            for top in range(2):
                full = ns | (top << (L - 1))
                old = full >> 1
                e = x[t] - out[full]
                mval = metric[old] + e * e
                if mval < best:
                    best = mval
                    bo = old
            new[ns] = best
            surv[t, ns] = bo
        mn = new.min()
        for ns in range(S):
            metric[ns] = new[ns] - mn
    st = np.argmin(metric)
    dec = np.empty(T)
    for t in range(T - 1, -1, -1):
        dec[t] = 1.0 if (st & 1) else -1.0
        st = surv[t, st]
    return dec


def mlse(x: np.ndarray, ch: Channel, block: int = 4096) -> np.ndarray:
    out = state_outputs(ch)
    L = ch.L
    if L == 1:
        return np.sign(x)
    D = 20 * L
    T = len(x)
    dec = np.empty(T)
    for a in range(0, T, block):
        b = min(T, a + block)
        lo, hi = max(0, a - D), min(T, b + D)
        d = _viterbi_block(x[lo:hi], out, L)
        dec[a:b] = d[a - lo:a - lo + (b - a)]
    return dec


def window_centres(ch: Channel, m: int, d: int):
    """Noiseless window vectors for every (m+L-1)-symbol state and the symbol s[n-d]."""
    L = ch.L
    K = m + L - 1
    k = np.arange(2**K)
    bits = (k[:, None] >> np.arange(K)[None, :]) & 1  # bit j -> s[n-j]
    s = 2.0 * bits - 1.0
    h = np.asarray(ch.h)
    C = np.stack([ch.f(s[:, i:i + L] @ h) for i in range(m)], axis=1)  # window x[n-i]
    return C, s[:, d], bits


def bayes_decide(X: np.ndarray, C: np.ndarray, cls: np.ndarray, sigma: float,
                 chunk: int = 4096) -> np.ndarray:
    """Log-likelihood-ratio of s[n-d]=+1 vs -1 given window X under Gaussian mixtures."""
    pos, neg = C[cls > 0], C[cls < 0]
    out = np.empty(len(X))
    inv = 1.0 / (2 * sigma * sigma)
    for a in range(0, len(X), chunk):
        xb = X[a:a + chunk]

        def lse(Cc):
            d2 = (xb**2).sum(1)[:, None] - 2 * xb @ Cc.T + (Cc**2).sum(1)[None, :]
            z = -d2 * inv
            mx = z.max(1, keepdims=True)
            return mx[:, 0] + np.log(np.exp(z - mx).sum(1))

        out[a:a + chunk] = lse(pos) - lse(neg)
    return out


class MAPWindow:
    name = "MAP"

    def __init__(self, ch: Channel, m: int, d: int, sigma: float):
        self.C, self.cls, _ = window_centres(ch, m, d)
        self.sigma, self.m = sigma, m

    def predict(self, X):
        return bayes_decide(X, self.C, self.cls, self.sigma)

    def macs(self):
        return self.C.shape[0] * (self.m + 1)  # plus one exp per centre


class RBFEqualiser:
    """Bayesian/RBF equaliser with centres learned from the training preamble.

    Each training window's channel state is known (training symbols are known), so the
    centre of each state is the mean of its received windows (supervised clustering).
    States never seen in training fall back to centres predicted from a least-squares
    channel estimate. Width = pooled residual standard deviation. This avoids the 1995
    failure mode (centres left empty by a short repeated preamble).
    """
    name = "RBF"

    def __init__(self, m: int, d: int, L: int):
        self.m, self.d, self.L = m, d, L

    def fit(self, X, s_full, idx):
        m, d, L = self.m, self.d, self.L
        K = m + L - 1
        # state index for row with time n: bit j -> s[n-j]
        ok = idx - (K - 1) >= 0
        X, idx = X[ok], idx[ok]
        st = np.zeros(len(idx), dtype=np.int64)
        for j in range(K):
            st |= (s_full[idx - j] > 0).astype(np.int64) << j
        # LS linear channel estimate (with bias) for unseen states
        S = np.stack([s_full[idx - j] for j in range(L)], 1)
        A = np.hstack([S, np.ones((len(S), 1))])
        coef, *_ = np.linalg.lstsq(A, X[:, 0], rcond=None)
        lin = Channel("est", tuple(coef[:L]))
        C, cls, _ = window_centres(lin, m, d)
        C = C + coef[L]
        sums = np.zeros_like(C)
        cnt = np.bincount(st, minlength=C.shape[0])
        np.add.at(sums, st, X)
        seen = cnt > 0
        means = np.where(seen[:, None], sums / np.maximum(cnt, 1)[:, None], C)
        resid = X - means[st]
        dof = np.sum(cnt[cnt > 1] - 1)
        if dof >= 10:  # unbiased within-state noise estimate
            var = np.sum(resid[cnt[st] > 1] ** 2) / (dof * m)
        else:  # fall back to the LS-fit residual
            var = np.mean((X[:, 0] - A @ coef) ** 2)
        self.sigma = float(np.sqrt(max(var, 1e-4)))
        k0 = 1.0  # shrink sparsely observed centres towards the LS-model prediction
        C = (sums + k0 * C) / (cnt[:, None] + k0)
        self.C, self.cls, self.frac_seen = C, cls, float(seen.mean())
        return self

    def predict(self, X):
        return bayes_decide(X, self.C, self.cls, self.sigma)

    def macs(self):
        return self.C.shape[0] * (self.m + 1)

    def n_params(self):
        return self.C.size + 1
