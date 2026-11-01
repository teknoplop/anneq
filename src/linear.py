"""Linear baselines: LMS / RLS transversal equalisers and an RLS-trained decision-feedback
equaliser. All have a bias input (needed for the nonlinear channel, harmless otherwise).

Training uses known symbols; at test time the weights are frozen (offline scenario) so the
comparison with the neural networks, which are also frozen, is like for like.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def _aug(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


@njit(cache=True)
def _lms(X, y, mu, w, normalised):
    mse = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        u = X[i]
        e = y[i] - np.dot(w, u)
        mse[i] = e * e
        step = mu / (1e-6 + np.dot(u, u)) if normalised else mu
        w += step * e * u
    return w, mse


@njit(cache=True)
def _rls(X, y, lam, delta, w):
    n = X.shape[1]
    P = np.eye(n) / delta
    mse = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        u = X[i]
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        e = y[i] - w @ u
        mse[i] = e * e
        w += k * e
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)  # keep P symmetric (long runs otherwise drift and diverge)
    return w, mse


class LMS:
    name = "LMS"

    def __init__(self, m, mu=0.05, normalised=True, passes=1):
        self.m, self.mu, self.norm, self.passes = m, mu, normalised, passes
        self.w = np.zeros(m + 1)

    def fit(self, X, y):
        A = _aug(X)
        for _ in range(self.passes):
            self.w, self.mse = _lms(A, y, self.mu, self.w, self.norm)
        return self

    def predict(self, X):
        return _aug(X) @ self.w

    def macs(self):
        return self.m + 1  # inference; NLMS update adds ~2(m+1)

    def n_params(self):
        return self.m + 1


class RLS(LMS):
    name = "RLS"

    def __init__(self, m, lam=0.999, delta=0.01):
        self.m, self.lam, self.delta = m, lam, delta
        self.w = np.zeros(m + 1)

    def fit(self, X, y):
        self.w, self.mse = _rls(_aug(X), y, self.lam, self.delta, self.w)
        return self


@njit(cache=True)
def _dfe_run(X, fb0, w, m, nb):
    # X: feedforward windows (N, m); decisions fed back for s[n-d-1..n-d-nb]
    N = X.shape[0]
    out = np.empty(N)
    past = fb0.copy()
    for i in range(N):
        z = w[m] * 1.0  # bias
        for j in range(m):
            z += w[j] * X[i, j]
        for j in range(nb):
            z += w[m + 1 + j] * past[j]
        out[i] = z
        dec = 1.0 if z >= 0 else -1.0
        for j in range(nb - 1, 0, -1):
            past[j] = past[j - 1]
        if nb > 0:
            past[0] = dec
    return out


class DFE:
    """Feedforward window of m samples, nb feedback taps on past decisions, RLS trained."""
    name = "DFE"

    def __init__(self, m, nb, lam=0.999, delta=0.01):
        self.m, self.nb, self.lam, self.delta = m, nb, lam, delta
        self.w = np.zeros(m + 1 + nb)

    def _fb(self, y):
        # feedback regressor from true past symbols: y[i-1], ..., y[i-nb]
        N = len(y)
        F = np.zeros((N, self.nb))
        for j in range(1, self.nb + 1):
            F[j:, j - 1] = y[:-j]
        return F

    def fit(self, X, y):
        A = np.hstack([_aug(X), self._fb(y)])
        self.w, self.mse = _rls(A, y, self.lam, self.delta, self.w)
        return self

    def predict(self, X, y_hist=None):
        fb0 = np.ones(self.nb) if y_hist is None else y_hist[::-1][: self.nb].copy()
        return _dfe_run(X, fb0, self.w, self.m, self.nb)

    def macs(self):
        return self.m + 1 + self.nb

    def n_params(self):
        return self.m + 1 + self.nb
