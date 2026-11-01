"""Host side of the FPGA backprop study (BPBC 2-6-6-1 batch backprop, 1995 ANNEQ).

Subcommands (run with: uv run --no-project --with numpy --with pyserial python gateware/py/anneq_fpga.py ...):
  lut OUT.hex                      tanh table for the engine and the C reference
  job OUT.bin [--trials --snr --ch --epochs --seed]   training sets + initial weights (shared by all runs)
  numpy JOB.bin                    numpy batch trainer (float64), timing + BER
  ber JOB.bin W.bin [--fixed]      BER of trained weights (float forward pass, 10000 bits, as BPBC.C)
  board JOB.bin OUT.bin --port P   train on the FPGA over the UART, timing incl. transfer
  lms                              numpy ADALINE/LMS throughput (src/anneq_numpy.py)

Data follow BPBC.C: PN preamble (5-bit register, taps 2 and 5, state kept across trials), channel
FIR with delay line starting at -1, Gaussian noise variance 10^(-SNR/10), B+1 samples for B patterns.
Inputs are quantised to Q2.14 (clipped to +-2) and every implementation trains on the same values.
"""
import argparse
import glob
import struct
import sys
import time
import pathlib

import numpy as np

NW = 67
CH = {"ch1": [1.0, 0.5], "ch2": [0.5, 1.0]}


def make_lut():
    i = np.arange(2048)
    return np.round(16384 * np.tanh((i + 0.5) / 512)).astype(int)


def gen_job(trials, snr, ch, epochs, seed, B=40, lr=0.1, mom=0.5):
    rng = np.random.default_rng(seed)
    h = np.array(CH[ch])
    sigma = np.sqrt(10.0 ** (-snr / 10.0))
    pn = [1, 1, 1, 1, 1]
    xs, ws = [], []
    for _ in range(trials):
        line = -np.ones(len(h))
        sig = 1.0
        x, d = np.empty(B + 1), np.empty(B + 1)
        for i in range(B + 1):
            line[0] = sig
            d[i] = sig
            x[i] = line @ h + sigma * rng.standard_normal()
            s = pn[1] ^ pn[4]
            pn = [s] + pn[:4]
            sig = 1.0 if s else -1.0
            line[1:] = line[:-1].copy()
        xq = np.clip(np.round(x * 16384), -32767, 32767).astype(np.int64)
        xs.append((xq & 0xFFFF) | ((d > 0).astype(np.int64) << 16))
        ws.append(np.round(0.3 * rng.random(NW) * 2 ** 28).astype(np.int64))
    K1, K2 = round(lr * (1 - mom) * 65536), round(mom * 65536)
    return dict(B=B, E=epochs, K1=K1, K2=K2, T=trials, xm=np.array(xs), w=np.array(ws), snr=snr, ch=ch)


def write_job(path, j):
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", j["B"], j["E"], j["K1"], j["K2"], j["T"]))
        for t in range(j["T"]):
            f.write(j["xm"][t].astype("<i4").tobytes())
            f.write(j["w"][t].astype("<i4").tobytes())
    pathlib.Path(str(path) + ".meta").write_text(f"{j['snr']} {j['ch']}\n")


def read_job(path):
    b = pathlib.Path(path).read_bytes()
    B, E, K1, K2, T = struct.unpack_from("<5i", b)
    a = np.frombuffer(b[20:], "<i4").reshape(T, B + 1 + NW).astype(np.int64)
    snr, ch = pathlib.Path(str(path) + ".meta").read_text().split()
    return dict(B=B, E=E, K1=K1, K2=K2, T=T, xm=a[:, :B + 1], w=a[:, B + 1:], snr=float(snr), ch=ch)


def decode(j):
    x = ((j["xm"] & 0xFFFF) ^ 0x8000) - 0x8000
    return x / 16384.0, np.where((j["xm"] >> 16) & 1, 1.0, -1.0)


def unpack(W):
    """W (T,67) float -> W1 (T,6,3), W2 (T,6,7), W3 (T,7)"""
    return W[:, :18].reshape(-1, 6, 3), W[:, 18:60].reshape(-1, 6, 7), W[:, 60:]


def train_numpy(j, lr=0.1, mom=0.5):
    """Idiomatic numpy: batch backprop, all trials and all B patterns of an epoch at once."""
    x, d = decode(j)
    T, B = j["T"], j["B"]
    A0 = np.stack([x[:, 1:], x[:, :-1], -np.ones((T, B))], -1)      # (T,B,3)
    D = d[:, 1:]
    W = j["w"] / 2.0 ** 28
    W1, W2, W3 = [a.copy() for a in unpack(W)]
    M1, M2, M3 = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(W3)
    ones = -np.ones((T, B, 1))
    for _ in range(j["E"]):
        A1 = np.concatenate([np.tanh(np.einsum("tbj,tnj->tbn", A0, W1)), ones], -1)
        A2 = np.concatenate([np.tanh(np.einsum("tbj,tnj->tbn", A1, W2)), ones], -1)
        y = np.tanh(np.einsum("tbj,tj->tb", A2, W3))
        d3 = (1 - y * y) * (D - y)
        d2 = d3[..., None] * W3[:, None, :6] * (1 - A2[..., :6] ** 2)
        d1 = np.einsum("tbk,tki->tbi", d2, W2[:, :, :6]) * (1 - A1[..., :6] ** 2)
        M3 = lr * (1 - mom) * np.einsum("tb,tbj->tj", d3, A2) + mom * M3
        M2 = lr * (1 - mom) * np.einsum("tbn,tbj->tnj", d2, A1) + mom * M2
        M1 = lr * (1 - mom) * np.einsum("tbn,tbj->tnj", d1, A0) + mom * M1
        W1 += M1; W2 += M2; W3 += M3
    return np.concatenate([W1.reshape(T, -1), W2.reshape(T, -1), W3], 1)


def forward(W, x0, x1):
    W1, W2, W3 = unpack(W)
    T = W.shape[0]
    a0 = np.stack([x0, x1, -np.ones_like(x0)], -1)
    a1 = np.concatenate([np.tanh(np.einsum("tbj,tnj->tbn", a0, W1)), -np.ones(x0.shape + (1,))], -1)
    a2 = np.concatenate([np.tanh(np.einsum("tbj,tnj->tbn", a1, W2)), -np.ones(x0.shape + (1,))], -1)
    return np.tanh(np.einsum("tbj,tj->tb", a2, W3))


def ber(j, W, bits=10000, seed=12345):
    """Per-trial BER over `bits` random bits (BPBC.C test loop), float forward pass."""
    rng = np.random.default_rng(seed)
    T = W.shape[0]
    h = np.array(CH[j["ch"]])
    sigma = np.sqrt(10.0 ** (-j["snr"] / 10.0))
    b = np.where(rng.random((T, bits + len(h))) > 0.5, 1.0, -1.0)
    sig = sum(h[k] * b[:, len(h) - 1 - k: len(h) - 1 - k + bits] for k in range(len(h)))
    r = sig + sigma * rng.standard_normal((T, bits))
    x0, x1 = r, np.concatenate([np.zeros((T, 1)), r[:, :-1]], 1)
    y = forward(W, x0, x1)
    return (np.sign(y + (y == 0)) != b[:, len(h) - 1:len(h) - 1 + bits]).mean(1)


def train_mse(j, W):
    x, d = decode(j)
    y = forward(W, x[:, 1:], x[:, :-1])
    return ((d[:, 1:] - y) ** 2).mean(1)


def read_w(path, T):
    return np.fromfile(path, "<i4").reshape(T, NW).astype(np.int64)


# ---------------------------------------------------------------- board protocol (gateware/rtl/top.v)
def board(j, out, port, baud, nengines):
    import serial
    s = serial.Serial(port, baud, timeout=30)
    s.reset_input_buffer()

    def wr(e, m, a, v):
        return b"W" + struct.pack("<BBHi", e, m, a, int(v))

    T, B = j["T"], j["B"]
    t0 = time.perf_counter()
    res = np.zeros((T, NW), np.int64)
    cycles, xfer = 0, 0.0
    s.write(wr(0, 3, 0, B) + wr(0, 3, 1, j["E"]) + wr(0, 3, 2, j["K1"]) + wr(0, 3, 3, j["K2"]))
    for base in range(0, T, nengines):
        grp = range(base, min(T, base + nengines))
        ta = time.perf_counter()
        buf = bytearray()
        for e, t in enumerate(grp):
            buf += b"".join(wr(e, 1, i, v) for i, v in enumerate(j["xm"][t]))
            buf += b"".join(wr(e, 0, i, v) for i, v in enumerate(j["w"][t]))
        buf += wr(0, 3, 4, (1 << len(grp)) - 1)       # engine enable mask
        s.write(bytes(buf))
        s.flush()
        tb = time.perf_counter()
        s.write(b"G")
        r = s.read(4)
        if len(r) != 4:
            sys.exit("board: no reply to G")
        tc = time.perf_counter()
        cycles += struct.unpack("<I", r)[0]
        q = b"".join(b"R" + struct.pack("<BH", e, i) for e in range(len(grp)) for i in range(NW))
        s.write(q)
        r = s.read(4 * NW * len(grp))
        if len(r) != 4 * NW * len(grp):
            sys.exit("board: short weight read")
        td = time.perf_counter()
        res[list(grp)] = np.frombuffer(r, "<i4").reshape(len(grp), NW)
        xfer += (tb - ta) + (td - tc)
    t1 = time.perf_counter()
    res.astype("<i4").tofile(out)
    return dict(wall=t1 - t0, cycles=cycles, xfer=xfer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("args", nargs="*")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--snr", type=float, default=10)
    p.add_argument("--ch", default="ch1")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--port", default=(glob.glob("/dev/cu.usbserial-*") or [None])[0])
    p.add_argument("--baud", type=int, default=2000000)
    p.add_argument("--engines", type=int, default=8)
    p.add_argument("--clk", type=float, default=50e6)
    a = p.parse_args()
    if a.cmd == "lut":
        pathlib.Path(a.args[0]).write_text("".join(f"{v:04x}\n" for v in make_lut()))
    elif a.cmd == "job":
        write_job(a.args[0], gen_job(a.trials, a.snr, a.ch, a.epochs, a.seed))
    elif a.cmd == "numpy":
        j = read_job(a.args[0])
        train_numpy(dict(j, E=2))                      # warm-up
        t0 = time.perf_counter()
        W = train_numpy(j)
        dt = time.perf_counter() - t0
        n = j["T"] * j["E"] * j["B"]
        print(f"numpy: {dt:.3f} s  {n/dt:.3e} presentations/s  {j['T']*j['E']/dt:.3e} epoch-updates/s")
        print(f"numpy BER mean {ber(j, W).mean():.3e}  train MSE mean {train_mse(j, W).mean():.2e}")
        if len(a.args) > 1:
            np.round(W * 2 ** 28).astype("<i4").tofile(a.args[1])
    elif a.cmd == "ber":
        j = read_job(a.args[0])
        W = read_w(a.args[1], j["T"]) / 2.0 ** 28
        b = ber(j, W)
        print(f"BER mean {b.mean():.3e} median {np.median(b):.3e}  train MSE mean {train_mse(j, W).mean():.2e}")
    elif a.cmd == "board":
        j = read_job(a.args[0])
        r = board(j, a.args[1], a.port, a.baud, a.engines)
        n = j["T"] * j["E"] * j["B"]
        comp = r["cycles"] / a.clk
        print(f"board: wall {r['wall']:.3f} s (transfer {r['xfer']:.3f} s), engine time {comp:.4f} s ({r['cycles']} cycles)")
        print(f"board: {n/comp:.3e} presentations/s on chip, {n/r['wall']:.3e} end to end")
    elif a.cmd == "lms":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
        import anneq_numpy as lms_np
        t0 = time.perf_counter()
        lms_np.run_lms([1.0, 0.5], 7, 0.01, 10, trials=100, test_bits=10, retrain=False)
        dt = time.perf_counter() - t0
        print(f"numpy LMS 7 taps, 100 trials x 1000 bits: {dt:.3f} s  {100*1000/dt:.3e} updates/s")
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
