"""Neural equalisers (PyTorch). All output a logit for s[n-d] = +1.

Window models (MLP, CNN, Transformer) see exactly the same m-sample window as the linear
equalisers and the MAP bound. The GRU is genuinely recurrent: it is fed one received sample
per step, x[n], carries its hidden state along the stream, and emits s[n-d] at step n, so
its look-ahead (decision delay d) is the same as everyone else's while its look-back is
unbounded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    kind = "window"

    def __init__(self, m, h1=32, h2=16):
        super().__init__()
        self.m, self.h1, self.h2 = m, h1, h2
        self.net = nn.Sequential(nn.Linear(m, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh(),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def macs(self):
        return self.m * self.h1 + self.h1 * self.h2 + self.h2


class CNN(nn.Module):
    kind = "window"

    def __init__(self, m, c=16, k=3):
        super().__init__()
        self.m, self.c, self.k = m, c, k
        self.conv = nn.Sequential(nn.Conv1d(1, c, k), nn.GELU(), nn.Conv1d(c, c, k), nn.GELU())
        self.L2 = m - 2 * (k - 1)
        self.head = nn.Linear(c * self.L2, 1)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))
        return self.head(z.flatten(1)).squeeze(-1)

    def macs(self):
        L1 = self.m - (self.k - 1)
        return L1 * self.k * self.c + self.L2 * self.k * self.c * self.c + self.c * self.L2


class TinyTransformer(nn.Module):
    """Each window sample is a token (value + learned position), 1 encoder layer, 2 heads."""
    kind = "window"

    def __init__(self, m, dm=16, heads=2, ff=32):
        super().__init__()
        self.m, self.dm, self.ff = m, dm, ff
        self.inp = nn.Linear(1, dm)
        self.pos = nn.Parameter(torch.randn(m, dm) * 0.1)
        self.enc = nn.TransformerEncoderLayer(dm, heads, ff, dropout=0.0, batch_first=True,
                                              norm_first=True)
        self.head = nn.Linear(dm * m, 1)

    def forward(self, x):
        z = self.inp(x.unsqueeze(-1)) + self.pos
        z = self.enc(z)
        return self.head(z.flatten(1)).squeeze(-1)

    def macs(self):
        m, d, f = self.m, self.dm, self.ff
        return m * d + m * 4 * d * d + 2 * m * m * d + m * 2 * d * f + m * d


class GRUEq(nn.Module):
    kind = "seq"

    def __init__(self, d, hidden=16):
        super().__init__()
        self.d, self.hdim = d, hidden
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h0=None):
        z, h = self.gru(x.unsqueeze(-1), h0)
        return self.head(z).squeeze(-1), h

    def macs(self):
        return 3 * (self.hdim + self.hdim * self.hdim) + self.hdim


def make(name, m, d):
    return {"MLP": lambda: MLP(m), "CNN": lambda: CNN(m), "TF": lambda: TinyTransformer(m),
            "GRU": lambda: GRUEq(d)}[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@dataclass
class TrainCfg:
    lr: float = 3e-3
    batch: int = 256
    max_steps: int = 3000
    eval_every: int = 200
    patience: int = 4
    seq_len: int = 96  # GRU training chunk
    burn: int = 16  # GRU warm-up steps excluded from the loss
    weight_decay: float = 1e-4


def _seq_batch(x, s, d, cfg, gen, B):
    T = len(x)
    starts = torch.randint(0, T - cfg.seq_len, (B,), generator=gen)
    ar = torch.arange(cfg.seq_len)
    xi = x[starts[:, None] + ar]
    # target at step t is s[t-d]
    ti = starts[:, None] + ar - d
    yi = s[ti.clamp(min=0)]
    mask = (ar >= cfg.burn)[None, :] & (ti >= 0)
    return xi, yi, mask


def train(model, train_data, val_data, d, cfg: TrainCfg, seed=0):
    """train_data/val_data: (X, y) windows for window models, (x, s) streams for the GRU.
    Returns dict with steps, symbols seen, wall time. Keeps the best-val weights."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    A, Y = (torch.as_tensor(a, dtype=torch.float32) for a in train_data)
    seq = model.kind == "seq"
    if seq:
        B = max(1, min(cfg.batch // 8, (len(A) - cfg.seq_len) // 4 or 1))
    else:
        B = min(cfg.batch, len(A))
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    step = 0
    for step in range(1, cfg.max_steps + 1):
        if seq:
            xb, yb, mask = _seq_batch(A, Y, d, cfg, gen, B)
            out, _ = model(xb)
            loss = (lossf(out, (yb > 0).float()) * mask).sum() / mask.sum()
        else:
            i = torch.randint(0, len(A), (B,), generator=gen)
            loss = lossf(model(A[i]), (Y[i] > 0).float()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if val_data is not None and step % cfg.eval_every == 0:
            v = val_loss(model, val_data, d, cfg)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    tokens = step * B * (cfg.seq_len if seq else 1)
    return dict(steps=step, train_time=time.time() - t0, examples_seen=tokens,
                val_loss=float(best))


@torch.no_grad()
def predict(model, data, d, chunk=4096, warm=64):
    """Window models: data = X (N, m) -> logits (N,).
    GRU: data = x stream (T,) -> logits for s[t-d] at every t (T,), run in overlapping chunks."""
    model.eval()
    if model.kind == "window":
        X = torch.as_tensor(data, dtype=torch.float32)
        out = torch.cat([model(X[a:a + 65536]) for a in range(0, len(X), 65536)])
        model.train()
        return out.numpy().astype(np.float64)
    x = np.asarray(data, dtype=np.float32)
    T = len(x)
    n = -(-T // chunk)
    pad = np.concatenate([np.zeros(warm, np.float32), x, np.zeros(n * chunk - T, np.float32)])
    segs = np.stack([pad[i * chunk:i * chunk + chunk + warm] for i in range(n)])
    out, _ = model(torch.from_numpy(segs))
    model.train()
    return out[:, warm:].reshape(-1)[:T].numpy().astype(np.float64)


def val_loss(model, val_data, d, cfg):
    lossf = nn.BCEWithLogitsLoss()
    if model.kind == "seq":
        x, s = val_data
        z = predict(model, x, d)
        z, t = torch.as_tensor(z[d:]), torch.as_tensor(s[: len(s) - d] > 0).double()
    else:
        X, y = val_data
        z, t = torch.as_tensor(predict(model, X, d)), torch.as_tensor(y > 0).double()
    return float(lossf(z, t))
