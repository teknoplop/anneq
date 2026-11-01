"""Line-for-line Python port of the four 1995 equaliser programs, stdlib only.

    anneq_orig.py PROG [TEST]      PROG = LMS, RNN, RB or BPBC

Run in a folder holding the program's config and WEIGHTS.TXT, exactly as the DOS programs:
  LMS   LMS_CON.TXT  -> ERRORS.TXT BER.TXT FAIL.TXT     (LMS.C + LMS_INIT.H, ADALINE/LMS)
  RNN   RNN_CON.TXT  -> ERRORS.TXT BER.TXT FAIL.TXT     (RNN.C + RNN_INIT.H, RTRL recurrent)
  RB    RB_CON.TXT   -> BER.TXT                         (RB.C + RB_INIT.H, radial basis)
  BPBC  e:\\temp\\bp_const.txt, e:\\temp\\weights.txt -> e:\\temp\\errors.txt, e:\\temp\\ber.txt
        (BPBC.C + BP_INIT.H, batch backprop; on a hosted OS these are literal file names in the
        current folder, as for the clang build in src/tools/clang)
Every 1995 bug is kept (see results/RESULTS.md); the corrected programs are anneq_fixed.py.
Screen output (gotoxy/cprintf) is reduced to plain progress lines on stderr. With the same
EQ_SEED, output files are byte-identical to the clang builds (src/tools/check_pyport.py).

From EQ_FUNCS.C: same names, same loops. Differences from C, by necessity:
- rand()/srand() are Borland C++ 3.1's LCG (as in src/openwatcom/eqcompat.c), so
  with the same seeds this port draws exactly the same numbers as the DOS and
  clang builds. EQ_SEED / EQ_RAND=libc behave as there (libc -> Python random).
- "far" pointers become Python lists of small objects; arrays are 0-based as in C.
- This is the memory-safe version: every array is a Python list, so any
  out-of-bounds index raises IndexError. The one overrun the 1995 code really
  commits (fill_rand in LMS, see results/RESULTS.md) is dropped, matching the
  harmless outcome; EQ_STRICT=1 raises instead.
- BPBC: the two unguarded debug weight dumps (print_weights with its layer arguments swapped,
  which reads the input layer's NULL weight pointers) and their "press a key" waits only affect
  the screen, so they are not ported; EQ_STRICT=1 raises where C would dereference NULL. The
  bp_tidyup use-after-free has no Python equivalent. Results match src/tools/clang/build_fix/bpbc
  (BP-DEBUG fix only) bit for bit with the same EQ_SEED.
"""
import math, os, sys, time, random as _pyrandom

RAND_MAX = 0x7FFF
_seed = 1
_seed_base = int(os.environ["EQ_SEED"]) if "EQ_SEED" in os.environ else -1
_seed_k = 0
STRICT = os.environ.get("EQ_STRICT") == "1"
_use_libc = os.environ.get("EQ_RAND", "").lower().startswith("l")


def rand():
    global _seed
    if _use_libc:
        return int(_pyrandom.random() * RAND_MAX)
    _seed = (_seed * 22695477 + 1) & 0xFFFFFFFF
    return (_seed >> 16) & 0x7FFF


def srand(s):
    global _seed, _seed_k
    if _seed_base >= 0:
        s = _seed_base + _seed_k
        _seed_k += 1
    _seed = s & 0xFFFFFFFF
    _pyrandom.seed(s)


def URAND():
    return rand() / RAND_MAX


SIG_A = 1.0
SIG_B = 2.0


def SIGMOID(x):
    return SIG_A * math.tanh(SIG_B * x * 0.5)


def DER_SIGMOID(x):
    return SIG_A * 0.5 * SIG_B * (1.0 - x * x)


class FIR_MODEL_STR:
    __slots__ = ("value", "weight")

    def __init__(self):
        self.value = 0.0
        self.weight = 0.0


class SENSE_STR:
    __slots__ = ("sense", "sum")

    def __init__(self):
        self.sense = 0.0
        self.sum = 0.0


class WEIGHT_STR:
    __slots__ = ("weight", "delta", "delta_sf", "sensitivity")

    def __init__(self):
        self.weight = 0.0
        self.delta = 0.0
        self.delta_sf = 0.0
        self.sensitivity = None


class NODE_STR:
    __slots__ = ("weights", "derivative", "output", "counter")

    def __init__(self):
        self.weights = None
        self.derivative = 0.0
        self.output = 0.0
        self.counter = 0.0


class ANN_STR:
    def __init__(self):
        self.nodes_0 = self.nodes_1 = self.nodes_2 = self.nodes_3 = 0
        self.layer_0 = self.layer_1 = self.layer_2 = self.layer_3 = None
        self.learn_rate = 0.0
        self.momentum = 0.0


def calloc(n, cls):
    return [cls() for _ in range(n)]


def params(no_pars, par_txt):
    test = 0
    if no_pars > 1:
        input_txt = par_txt[1].upper()
        if "TEST" == input_txt:
            test = 1
    return test


def noise(sig_to_noise):
    while True:
        rand1 = URAND()
        if rand1 != 0.0:
            break
    return (math.sqrt(-2.0 * math.pow(10.0, (-sig_to_noise / 10.0)) * math.log(rand1))
            # C: cos(2.0 * M_PI * URAND) with the unbracketed URAND macro is
            # (2*pi*rand())/RAND_MAX, which rounds differently from 2*pi*(rand()/RAND_MAX)
            * math.cos(2.0 * math.pi * rand() / RAND_MAX))


def fir_output(filter, fir_length):
    prod = 0.0
    for i in range(fir_length):
        prod += filter[i].value * filter[i].weight
    return prod


def shift_right_fir(filter, fir_length):
    for i in range(fir_length - 1, 0, -1):
        filter[i].value = filter[i - 1].value


def sign(value):
    result = 1
    if value < 0.0:
        result = -1
    return result


def update_weights_sf(layer1, layer2, nodes1, nodes2):
    for i in range(nodes2):
        for j in range(nodes1):
            layer2[i].weights[j].delta_sf += layer2[i].derivative * layer1[j].output


def update_weights_fi(layer2, nodes1, nodes2, lr, mc):
    for i in range(nodes2):
        for j in range(nodes1):
            w = layer2[i].weights[j]
            w.delta = w.delta_sf * lr * (1.0 - mc) + mc * w.delta
            w.weight += w.delta
            w.delta_sf = 0.0


def derivatives(layer1, layer2, nodes1, nodes2):
    for i in range(nodes1):
        sum_ = 0.0
        for j in range(nodes2):
            sum_ += layer2[j].weights[i].weight * layer2[j].derivative
        layer1[i].derivative = sum_ * DER_SIGMOID(layer1[i].output)


def ffwd_sig(inputs, nodes, no_inputs, no_nodes):
    for i in range(no_nodes):
        input_ = 0.0
        for j in range(no_inputs):
            input_ += inputs[j].output * nodes[i].weights[j].weight
        nodes[i].output = SIGMOID(input_)


def get_state(data, length):
    state = 0
    power = int(math.pow(2, length - 1))
    for i in range(length):
        if data[i].value == 1:
            state += power
        power //= 2
    return state


def dist(inputs, nodes, no_inputs, no_nodes, width):
    for i in range(no_nodes):
        distance = 0.0
        for j in range(no_inputs):
            distance += ((nodes[i].weights[j].weight - inputs[j].output)
                         * (nodes[i].weights[j].weight - inputs[j].output))
        x = math.sqrt(distance)
        nodes[i].output = math.exp(-(x * x) / width)   # RAD_BASIS


def ffwd_ada(inputs, nodes, no_inputs, no_nodes):
    for i in range(no_nodes):
        input_ = 0.0
        for j in range(no_inputs):
            input_ += inputs[j].output * nodes[i].weights[j].weight
        nodes[i].output = input_


def next_pn(pn_shift):
    signal = float(pn_shift[1] ^ pn_shift[4])
    for i in range(4, 0, -1):
        pn_shift[i] = pn_shift[i - 1]
    pn_shift[0] = int(signal)
    if signal == 0:
        signal = -1.0
    return signal


def shift_right_layer(layer, no_nodes):
    for i in range(no_nodes - 1, 0, -1):
        layer[i].output = layer[i - 1].output


def save_doubles(list_, length, div, path, name):
    filename = "%s%s" % (path, name)
    try:
        output_file = open(filename, "w")
    except OSError:
        print("File %s cannot be created" % name)
        sys.exit(1)
    for i in range(length):
        output_file.write("%+2.7f\n" % (list_[i] / div))
    output_file.close()


def fill_rand(layer_2, nodes_1, nodes_2):
    for i in range(nodes_2):
        for j in range(nodes_1):
            v = URAND() * 0.1          # drawn even when j is out of range, as in C
            if j < len(layer_2[i].weights):
                layer_2[i].weights[j].weight = v
            elif STRICT:               # EQ_STRICT=1: report the 1995 overrun instead of dropping it
                raise IndexError("fill_rand: weight %d written past array of %d (heap overrun in C)"
                                 % (j, len(layer_2[i].weights)))

ann = ANN_STR()
pn_shift = [1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# LMS
modelline = delayline = errors = error_run = None
snr = pass_threshold = 0.0
taps = m_taps = bits = runs = x = y = delay = 0


def lms_init(path):
    global m_taps, bits, snr, runs, delay, modelline, delayline, errors, error_run
    try:
        with open("%sLMS_CON.TXT" % path) as f:
            v = f.read().split()
    except OSError:
        print("\n\nFile %sLMS_CON.TXT not found" % path)
        sys.exit(1)
    m_taps, bits, snr, runs, delay, ann.nodes_0, ann.learn_rate = (
        int(v[0]), int(v[1]), float(v[2]), int(v[3]), int(v[4]), int(v[5]), float(v[6]))
    ann.nodes_1 = 1
    modelline = calloc(m_taps, FIR_MODEL_STR)
    with open("%sWEIGHTS.TXT" % path) as f:
        w = f.read().split()
    for i in range(m_taps):
        modelline[i].weight = float(w[i])
    ann.layer_0 = calloc(ann.nodes_0 + 1, NODE_STR)
    ann.layer_1 = calloc(1, NODE_STR)
    ann.layer_1[0].weights = calloc(ann.nodes_0 + 1, WEIGHT_STR)
    errors = [0.0] * bits
    error_run = [0.0] * bits
    delayline = calloc(delay + 1, FIR_MODEL_STR)


def lms_main(no_pars, par_txt):
    ber = 0.0
    failiures = 0.0
    path = ""
    lms_init(path)
    test = params(no_pars, par_txt)

    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        fail = 1
        while fail:
            srand(int(time.time()) & 0xFFFF)

            for i in range(m_taps):
                modelline[i].value = -1.0
            for i in range(ann.nodes_0):
                ann.layer_0[i].output = -1.0

            fill_rand(ann.layer_1, ann.nodes_0 + ann.nodes_1 + 1, ann.nodes_1)

            signal = 1.0
            delay_count = delay
            mean_err = 0.0

            for bit_no in range(bits):
                delayline[0].value = modelline[0].value = signal
                ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr)

                if delay_count == 0:
                    ffwd_ada(ann.layer_0, ann.layer_1, ann.nodes_0 + 1, ann.nodes_1)
                    error = delayline[delay].value - ann.layer_1[0].output
                    error_run[bit_no - delay] = error * error
                    if bit_no >= bits - 400:
                        mean_err += error * error
                    for j in range(ann.nodes_0 + 1):
                        ann.layer_1[0].weights[j].weight += (
                            2.0 * ann.learn_rate * error * ann.layer_0[j].output)
                else:
                    delay_count -= 1

                signal = next_pn(pn_shift)
                shift_right_layer(ann.layer_0, ann.nodes_0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)

            if (mean_err / 400) < 0.6:
                fail = 0
                sys.stderr.write(" %1.3f PASSED" % (mean_err / 400))
            else:
                sys.stderr.write(" %1.3f FAILED" % (mean_err / 400))
                failiures += 1

        if not fail:
            for i in range(bits - delay):
                errors[i] += error_run[i]

        if test:
            no_errors = 0
            for i in range(10000):
                delayline[0].value = modelline[0].value = sign((2.0 * URAND()) - 1.0)
                ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr)
                ffwd_ada(ann.layer_0, ann.layer_1, ann.nodes_0 + 1, ann.nodes_1)
                if sign(ann.layer_1[0].output) != delayline[delay].value:
                    no_errors += 1
                shift_right_layer(ann.layer_0, ann.nodes_0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)
            sys.stderr.write(" ERRORS:%05d\n" % no_errors)
            ber += no_errors / 10000

    save_doubles(errors, bits - delay, runs, path, "ERRORS.TXT")
    save_doubles([ber], 1, runs, path, "BER.TXT")
    save_doubles([failiures], 1, runs, path, "FAIL.TXT")


# ---------------------------------------------------------------------------
# RNN
def KRONECKER(x, y):
    return float(x == y)


def rnn_main(no_pars, par_txt):
    path = ""
    with open("%sRNN_CON.TXT" % path) as f:
        v = f.read().split()
    m_taps, bits, snr, runs, delay = int(v[0]), int(v[1]), float(v[2]), int(v[3]), int(v[4])
    ann.nodes_0, ann.nodes_1, ann.learn_rate = int(v[5]), int(v[6]), float(v[7])
    modelline = calloc(m_taps, FIR_MODEL_STR)
    with open("%sWEIGHTS.TXT" % path) as f:
        w = f.read().split()
    for i in range(m_taps):
        modelline[i].weight = float(w[i])
    n0, n1 = ann.nodes_0, ann.nodes_1
    ann.layer_0 = calloc(n0 + n1 + 1, NODE_STR)
    ann.layer_1 = calloc(n1, NODE_STR)
    for i in range(n1):
        ann.layer_1[i].weights = calloc(n0 + n1 + 1, WEIGHT_STR)
        for j in range(n0 + n1 + 1):
            ann.layer_1[i].weights[j].sensitivity = calloc(n1, SENSE_STR)
    errors = [0.0] * bits
    error_run = [0.0] * bits
    delayline = calloc(delay + 1, FIR_MODEL_STR)
    L0, L1 = ann.layer_0, ann.layer_1

    ber = 0.0
    failiures = 0.0
    test = params(no_pars, par_txt)

    n_in = n0 + 1   # 1995 bug RNN-FFWD: feedback inputs not fed forward
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        fail = 1
        while fail:
            srand(int(time.time()) & 0xFFFF)
            for i in range(m_taps):
                modelline[i].value = -1.0
            for k in range(n1):
                for l in range(n0 + n1 + 1):
                    for j in range(n1):
                        L1[k].weights[l].sensitivity[j].sense = 0.0
            fill_rand(L1, n0 + n1 + 1, n1)
            L0[n0].output = -1.0
            signal = 1.0
            delay_count = delay
            mean_err = 0.0
            for bit_no in range(bits):
                delayline[0].value = modelline[0].value = signal
                L0[0].output = fir_output(modelline, m_taps) + noise(snr)
                if delay_count == 0:
                    for j in range(n1):
                        L0[n0 + 1 + j].output = L1[j].output
                    ffwd_sig(L0, L1, n_in, n1)
                    error = delayline[delay].value - L1[0].output
                    error_run[bit_no - delay] = error * error
                    if bit_no >= bits - 400:
                        mean_err += error * error
                    for k in range(n1):
                        for l in range(n0 + n1 + 1):
                            for j in range(n1):
                                s = L1[k].weights[l].sensitivity[j]
                                s.sum = 0.0
                                for i in range(n0 + 1, n0 + 1 + n1):
                                    s.sum += (L1[j].weights[i].weight
                                              * L1[k].weights[l].sensitivity[i - n0 - 1].sense
                                              + KRONECKER(k, j) * L0[l].output)
                    for k in range(n1):
                        for l in range(n0 + n1 + 1):
                            for j in range(n1):
                                s = L1[k].weights[l].sensitivity[j]
                                s.sense = DER_SIGMOID(L1[j].output) * (s.sum)
                    for k in range(n1):
                        for l in range(n1 + n0 + 1):
                            L1[k].weights[l].weight += (ann.learn_rate * error
                                                        * L1[k].weights[l].sensitivity[0].sense)
                else:
                    delay_count -= 1
                signal = next_pn(pn_shift)
                shift_right_layer(L0, n0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)
            if (mean_err / 400) < 0.6:
                fail = 0
            else:
                failiures += 1
        if not fail:
            for i in range(bits - delay):
                errors[i] += error_run[i]
        if test:
            no_errors = 0
            for i in range(10000):
                delayline[0].value = modelline[0].value = sign((2.0 * URAND()) - 1.0)
                L0[0].output = fir_output(modelline, m_taps) + noise(snr)
                for j in range(n1):
                    L0[n0 + 1 + j].output = L1[j].output
                ffwd_sig(L0, L1, n_in, n1)
                if sign(L1[0].output) != delayline[delay].value:
                    no_errors += 1
                shift_right_layer(L0, n0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)
            sys.stderr.write(" ERRORS:%05d\n" % no_errors)
            ber += no_errors / 10000
    save_doubles(errors, bits - delay, runs, path, "ERRORS.TXT")
    save_doubles([ber], 1, runs, path, "BER.TXT")
    save_doubles([failiures], 1, runs, path, "FAIL.TXT")


# ---------------------------------------------------------------------------
# RB
def rb_main(no_pars, par_txt):
    path = ""
    with open("%sRB_CON.TXT" % path) as f:
        v = f.read().split()
    m_taps, bits, snr, runs = int(v[0]), int(v[1]), float(v[2]), int(v[3])
    ann.nodes_0, ann.nodes_1, ann.learn_rate = int(v[4]), int(v[5]), float(v[6])
    ann.nodes_2 = 1
    modelline = calloc(m_taps, FIR_MODEL_STR)
    with open("%sWEIGHTS.TXT" % path) as f:
        w = f.read().split()
    for i in range(m_taps):
        modelline[i].weight = float(w[i])
    ann.layer_0 = calloc(ann.nodes_0, NODE_STR)
    ann.layer_1 = calloc(ann.nodes_1, NODE_STR)
    ann.layer_2 = calloc(1, NODE_STR)
    for i in range(ann.nodes_1):
        ann.layer_1[i].weights = calloc(ann.nodes_0, WEIGHT_STR)
    ann.layer_2[0].weights = calloc(ann.nodes_1, WEIGHT_STR)
    for i in range(ann.nodes_1 // 2):
        ann.layer_2[0].weights[i].weight = -1.0
    i = ann.nodes_1 // 2
    while i < ann.nodes_1:            # for(i=i; ...) in RB_INIT.H
        ann.layer_2[0].weights[i].weight = 1.0
        i += 1
    memory = int(math.log10(ann.nodes_1) / math.log10(2))
    delayline = calloc(memory, FIR_MODEL_STR)

    ber = 0.0
    test = params(no_pars, par_txt)
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        srand(int(time.time()) & 0xFFFF)
        for i in range(ann.nodes_1):
            for j in range(ann.nodes_0):
                ann.layer_1[i].weights[j].weight = 0.0
        for i in range(m_taps):
            modelline[i].value = -1.0
        for i in range(ann.nodes_0):
            ann.layer_0[i].output = -1.0
        for i in range(ann.nodes_1):
            ann.layer_1[i].counter = 0
        signal = 1.0
        delay_count = ann.nodes_0
        for bit_no in range(bits):
            delayline[0].value = modelline[0].value = signal
            ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr)
            if delay_count == 0:
                index = get_state(delayline, memory)
                node = ann.layer_1[index]
                for j in range(ann.nodes_0):
                    node.weights[j].weight *= node.counter
                    node.weights[j].weight += ann.layer_0[j].output
                node.counter += 1
                for j in range(ann.nodes_0):
                    node.weights[j].weight /= node.counter
            else:
                delay_count -= 1
            signal = next_pn(pn_shift)
            shift_right_layer(ann.layer_0, ann.nodes_0)
            shift_right_fir(modelline, m_taps)
            shift_right_fir(delayline, memory)
        if test:
            no_errors = 0
            for i in range(10000):
                modelline[0].value = sign((2.0 * URAND()) - 1.0)
                ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr)
                dist(ann.layer_0, ann.layer_1, ann.nodes_0, ann.nodes_1, ann.learn_rate)
                ffwd_ada(ann.layer_1, ann.layer_2, ann.nodes_1, ann.nodes_2)
                if sign(ann.layer_2[0].output) != modelline[0].value:
                    no_errors += 1
                shift_right_layer(ann.layer_0, ann.nodes_0)
                shift_right_fir(modelline, m_taps)
            sys.stderr.write(" LAST BER:%05d\n" % no_errors)
            ber += no_errors / 10000.0
    save_doubles([ber], 1, runs, path, "BER.TXT")


# ---------------------------------------------------------------------------
# BPBC
def bpbc_main(no_pars, par_txt):
    with open("e:\\temp\\bp_const.txt") as f:
        v = f.read().split()
    m_taps, bits, epochs, snr, runs = int(v[0]), int(v[1]), int(v[2]), float(v[3]), int(v[4])
    ann.nodes_0, ann.nodes_1, ann.nodes_2 = int(v[5]), int(v[6]), int(v[7])
    ann.learn_rate, ann.momentum = float(v[8]), float(v[9])
    ann.nodes_3 = 1
    modelline = calloc(m_taps, FIR_MODEL_STR)
    with open("e:\\temp\\weights.txt") as f:
        w = f.read().split()
    for i in range(m_taps):
        modelline[i].weight = float(w[i])
    ann.layer_0 = calloc(ann.nodes_0 + 1, NODE_STR)
    ann.layer_1 = calloc(ann.nodes_1 + 1, NODE_STR)
    ann.layer_2 = calloc(ann.nodes_2 + 1, NODE_STR)
    ann.layer_3 = calloc(ann.nodes_3, NODE_STR)
    for i in range(ann.nodes_1):
        ann.layer_1[i].weights = calloc(ann.nodes_0 + 1, WEIGHT_STR)
    for i in range(ann.nodes_2):
        ann.layer_2[i].weights = calloc(ann.nodes_1 + 1, WEIGHT_STR)
    ann.layer_3[0].weights = calloc(ann.nodes_2 + 1, WEIGHT_STR)
    errors = [0.0] * epochs
    half = int(math.ceil(ann.nodes_0 / 2.0))
    train_vect = [[0.0, 0.0] for _ in range(bits + half)]   # [value, desired]
    L0, L1, L2, L3 = ann.layer_0, ann.layer_1, ann.layer_2, ann.layer_3
    n0, n1, n2, n3 = ann.nodes_0, ann.nodes_1, ann.nodes_2, ann.nodes_3

    ber = 0.0
    test = params(no_pars, par_txt)
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        srand(int(time.time()) & 0xFFFF)
        for i in range(m_taps):
            modelline[i].value = -1.0
        signal = 1.0
        i = 0
        while i < (bits + math.ceil(n0 / 2.0)):
            train_vect[i][1] = modelline[0].value = signal
            train_vect[i][0] = fir_output(modelline, m_taps) + noise(snr)
            signal = next_pn(pn_shift)
            shift_right_fir(modelline, m_taps)
            i += 1

        if STRICT:
            raise RuntimeError("BPBC debug print_weights(): layer arguments swapped, reads NULL "
                               "weights of the input layer (crashes on a hosted OS)")

        for i in range(n1):
            for j in range(n0 + 1):
                ww = L1[i].weights[j]
                ww.weight = 0.3 * rand() / RAND_MAX; ww.delta = 0.0; ww.delta_sf = 0.0
        for i in range(n2):
            for j in range(n1 + 1):
                ww = L2[i].weights[j]
                ww.weight = 0.3 * rand() / RAND_MAX; ww.delta = 0.0; ww.delta_sf = 0.0
        for i in range(n3):
            for j in range(n2 + 1):
                ww = L3[i].weights[j]
                ww.weight = 0.3 * rand() / RAND_MAX; ww.delta = 0.0; ww.delta_sf = 0.0
        L0[n0].output = -1.0
        L1[n1].output = -1.0
        L2[n2].output = -1.0

        for i in range(epochs):
            sum_sqr_err = 0.0
            index = half
            for j in range(bits):
                for k in range(n0):
                    L0[k].output = train_vect[index - k][0]
                ffwd_sig(L0, L1, n0 + 1, n1)
                ffwd_sig(L1, L2, n1 + 1, n2)
                ffwd_sig(L2, L3, n2 + 1, n3)
                error = train_vect[index][1] - L3[0].output
                sum_sqr_err += error * error
                L3[0].derivative = DER_SIGMOID(L3[0].output) * error
                derivatives(L2, L3, n2, n3)
                derivatives(L1, L2, n1, n2)
                update_weights_sf(L2, L3, n2 + 1, n3)
                update_weights_sf(L1, L2, n1 + 1, n2)
                update_weights_sf(L0, L1, n0 + 1, n1)
                index += 1
            errors[i] += sum_sqr_err / bits
            update_weights_fi(L3, n2 + 1, n3, ann.learn_rate, ann.momentum)
            update_weights_fi(L2, n1 + 1, n2, ann.learn_rate, ann.momentum)
            update_weights_fi(L1, n0 + 1, n1, ann.learn_rate, ann.momentum)

        if test:
            no_errors = 0
            for i in range(10000):
                desired = modelline[0].value = sign(2.0 * URAND() - 1.0)
                L0[0].output = fir_output(modelline, m_taps) + noise(snr)
                ffwd_sig(L0, L1, n0 + 1, n1)
                ffwd_sig(L1, L2, n1 + 1, n2)
                ffwd_sig(L2, L3, n2 + 1, n3)
                if sign(L3[0].output) != desired:
                    no_errors += 1
                for j in range(n0 - 1, 0, -1):
                    L0[j].output = L0[j - 1].output
                shift_right_fir(modelline, m_taps)
            sys.stderr.write(" LAST BER:%05d\n" % no_errors)
            ber += no_errors / 10000

    with open("e:\\temp\\errors.txt", "w") as f:
        for i in range(epochs):
            f.write("%+2.15f\n" % (errors[i] / runs))
    with open("e:\\temp\\ber.txt", "w") as f:
        f.write("%+2.15f\n" % (ber / runs))


PROGRAMS = {"LMS": lms_main, "RNN": rnn_main, "RB": rb_main, "BPBC": bpbc_main}


def cli(programs):
    if len(sys.argv) < 2 or sys.argv[1].upper() not in programs:
        sys.exit("usage: %s {%s} [TEST]" % (os.path.basename(sys.argv[0]), ",".join(programs)))
    argv = sys.argv[1:]          # argv[1] is TEST, as in the DOS program's own command line
    programs[argv[0].upper()](len(argv), argv)


if __name__ == "__main__":
    cli(PROGRAMS)
