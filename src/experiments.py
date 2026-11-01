"""The two scenarios.

offline:  2026 deployment model. Each method is trained once on plenty of random symbols at
          the operating SNR (separate train / validation / test streams), then frozen and
          tested until it has made >= min_errors errors (or the bit cap is reached).
preamble: 1995 deployment model. Each method must adapt from scratch on an N-symbol PN
          preamble (non-repeating segment of a long m-sequence) and is then frozen and tested.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

import linear, neural, trellis
from channels import CHANNELS, mmse_delay, transmit, windows
from data import pn_sequence, rng_for
from metrics import clopper_pearson, wilson

LINEAR = ["LMS", "RLS", "DFE"]
NEURAL = ["MLP", "CNN", "TF", "GRU"]


@dataclass
class Setup:
    window: dict = field(default_factory=lambda: {"ch1": 7, "ch2": 7, "ch1nl": 7, "ch3": 15})
    delay_snr_db: float = 12.0
    max_states_bayes: int = 4096  # MAP/RBF only where 2^(m+L-1) is tractable
    lms_mu: float = 0.05  # NLMS step
    lms_passes_offline: int = 3


def geometry(chname, setup: Setup):
    ch = CHANNELS[chname]
    m = setup.window[chname]
    d = mmse_delay(ch, m, setup.delay_snr_db)
    return ch, m, d


def bayes_ok(ch, m, setup):
    return 2 ** (m + ch.L - 1) <= setup.max_states_bayes


def _fit_all(ch, m, d, s, x, setup, methods, cfg, seed, val=None, lms_passes=1):
    X, y, idx = windows(x, s, m, d)
    models, info = {}, {}
    for name in methods:
        t0 = time.time()
        extra = {}
        if name == "LMS":
            mdl = linear.LMS(m, setup.lms_mu, passes=lms_passes).fit(X, y)
        elif name == "RLS":
            mdl = linear.RLS(m).fit(X, y)
        elif name == "DFE":
            mdl = linear.DFE(m, ch.L - 1).fit(X, y)
        elif name == "RBF":
            mdl = trellis.RBFEqualiser(m, d, ch.L).fit(X, s, idx)
            extra["frac_centres_seen"] = mdl.frac_seen
        elif name in NEURAL:
            mdl = neural.make(name, m, d)
            if mdl.kind == "seq":
                tr = (x, s)
                va = None if val is None else (val[1], val[0])
            else:
                tr = (X, y)
                va = None if val is None else windows(val[1], val[0], m, d)[:2]
            extra.update(neural.train(mdl, tr, va, d, cfg, seed=seed))
        else:
            raise ValueError(name)
        info[name] = dict(train_time=extra.pop("train_time", time.time() - t0), **extra)
        models[name] = mdl
    return models, info


def _decide(name, mdl, x, s, m, d, ch):
    """Return (soft outputs, true symbols) for the common evaluation rows."""
    X, y, idx = windows(x, s, m, d)
    if name == "MLSE":
        return trellis.mlse(x, ch)[idx - d], y
    if name in NEURAL and mdl.kind == "seq":
        return neural.predict(mdl, x, d)[idx], y
    if name in NEURAL:
        return neural.predict(mdl, X, d), y
    return mdl.predict(X), y


def _meta(name, mdl, ch, m):
    if name == "MLSE":
        return dict(params=0, macs=2 * 2 ** (ch.L - 1) * 2)  # 2 branches/state, err^2 + add
    if name == "MAP":
        return dict(params=0, macs=mdl.macs())
    if name in NEURAL:
        return dict(params=neural.n_params(mdl), macs=mdl.macs())
    return dict(params=mdl.n_params(), macs=mdl.macs())


def _row(base, name, k, n, meta, info):
    lo, hi = clopper_pearson(k, n)
    wl, wh = wilson(k, n)
    return dict(base, method=name, errors=k, bits=n, ber=k / n, cp_lo=lo, cp_hi=hi,
                wilson_lo=wl, wilson_hi=wh, **meta, **info)


def offline(chname, snr, n_train=100_000, n_val=20_000, block=200_000, min_errors=100,
            max_bits=2_000_000, setup: Setup | None = None, cfg: neural.TrainCfg | None = None,
            methods=None):
    setup = setup or Setup()
    cfg = cfg or neural.TrainCfg()
    torch.set_num_threads(1)
    ch, m, d = geometry(chname, setup)
    methods = methods or (LINEAR + (["RBF"] if bayes_ok(ch, m, setup) else []) + NEURAL)
    rs = transmit(ch, n_train, snr, rng_for("off-train", chname, int(snr * 10)))
    rv = transmit(ch, n_val, snr, rng_for("off-val", chname, int(snr * 10)))
    models, info = _fit_all(ch, m, d, *rs, setup, methods, cfg, seed=int(snr * 10), val=rv,
                            lms_passes=setup.lms_passes_offline)
    sigma = ch.sigma(snr)
    refs = ["MLSE"] + (["MAP"] if bayes_ok(ch, m, setup) else [])
    if "MAP" in refs:
        models["MAP"] = trellis.MAPWindow(ch, m, d, sigma)
    models["MLSE"] = None
    errs = {k: 0 for k in models}
    bits = {k: 0 for k in models}
    rng = rng_for("off-test", chname, int(snr * 10))
    while True:
        todo = [k for k in models if errs[k] < min_errors and bits[k] < max_bits]
        if not todo:
            break
        s, x = transmit(ch, block, snr, rng)
        for k in todo:
            z, y = _decide(k, models[k], x, s, m, d, ch)
            errs[k] += int(np.count_nonzero((z >= 0) != (y > 0)))
            bits[k] += len(y)
    base = dict(scenario="offline", channel=chname, snr_db=snr, window=m, delay=d,
                n_train=n_train)
    return [_row(base, k, errs[k], bits[k], _meta(k, models[k], ch, m), info.get(k, {}))
            for k in models]


def preamble(chname, snr, n_pre, trial, n_test=50_000, setup: Setup | None = None,
             steps=1500, methods=None):
    setup = setup or Setup()
    torch.set_num_threads(1)
    ch, m, d = geometry(chname, setup)
    methods = methods or (LINEAR + (["RBF"] if bayes_ok(ch, m, setup) else []) + NEURAL)
    rng = rng_for("pre", chname, int(snr * 10), n_pre, trial)
    off = int(rng.integers(0, 2**15 - 1))
    pn = np.roll(pn_sequence(15), -off)[:n_pre]  # one non-repeating PN segment
    s, x = transmit(ch, n_pre, snr, rng, symbols=pn)
    cfg = neural.TrainCfg(max_steps=steps, eval_every=10**9,
                          seq_len=max(8, min(96, n_pre // 2)), burn=min(16, n_pre // 8))
    if n_pre - cfg.seq_len < 2:
        methods = [k for k in methods if k != "GRU"]
    models, info = _fit_all(ch, m, d, s, x, setup, methods, cfg, seed=trial, val=None,
                            lms_passes=1)
    st, xt = transmit(ch, n_test, snr, rng_for("pre-test", chname, int(snr * 10), trial))
    base = dict(scenario="preamble", channel=chname, snr_db=snr, window=m, delay=d,
                n_train=n_pre, trial=trial)
    rows = []
    for k, mdl in models.items():
        z, y = _decide(k, mdl, xt, st, m, d, ch)
        e = int(np.count_nonzero((z >= 0) != (y > 0)))
        rows.append(_row(base, k, e, len(y), _meta(k, mdl, ch, m), info[k]))
    return rows


__all__ = ["Setup", "offline", "preamble", "geometry", "asdict"]
