"""The four 1995 equaliser programs with the corrections of issues #10-#18.

    anneq_fixed.py PROG [TEST]      PROG = LMS, RNN, RB or BPBC

Same folders, config and output files as anneq_orig.py (the faithful port); unchanged helpers
are imported from there. Every fix is on by default (what was EQ_FIXED=1). Select fixes with
    EQ_FIX=TAG,TAG..   only these (e.g. the C ones, to compare with the src/tools/clang builds)
    EQ_NOFIX=TAG,..    all but these
Tags with a counterpart in src/fixes/*.patch give bit-identical results to the matching
src/tools/clang/build_fix* binary; the others have no C counterpart.
  LMS-FILL       #12 fill_rand fills nodes_0+1 weights, not one past the array      (LMS)
  LMS-BIAS       #13 bias input -1 as in Fig 5.1                                    (LMS)
  RNN-FFWD       #10 feedback inputs fed forward                                    (RNN)
  RNN-KRON       #11 /nodes_1 inside the sensitivity sum, as in C                   (RNN)
  BP-DEBUG           no debug dump of NULL weights                                  (BPBC)
  RETRY-CAP      #14 at most EQ_MAX_ATTEMPTS (10) attempts per trial; GAVEUP.TXT = share of
                     trials that never passed (their BER still counts, their curve does not),
                     ERRORS_ALL.TXT = curve over all trials                         (LMS, RNN)
  SEED-ONCE      #15 srand() once at start (EQ_SEED, default 1), never from the clock per trial
  RAND-PREAMBLE  #16 random +-1 training bits instead of the repeated 31-bit PN pattern
  BP-PATHS       #17 BPBC reads/writes BP_CONST.TXT, WEIGHTS.TXT, ERRORS.TXT, BER.TXT in the
                     run folder, not e:\\temp\\
  MINOR          #18 explicit exit status 0
"""
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anneq_orig import *  # noqa: F401,F403  unchanged helpers (rand, noise, fill_rand, ...)
from anneq_orig import cli

C_FIXES = ("LMS-FILL", "LMS-BIAS", "RNN-FFWD", "RNN-KRON", "BP-DEBUG")
PY_FIXES = (
    "RETRY-CAP",     # #14 at most MAX_ATTEMPTS training attempts per trial; GAVEUP.TXT = share of
                     #     trials that never passed (their BER still counts, their curve does not)
    "SEED-ONCE",     # #15 srand() once at start (EQ_SEED, default 1), never from the clock per trial
    "RAND-PREAMBLE", # #16 random +-1 training bits instead of the repeated 31-bit PN pattern
    "BP-PATHS",      # #17 BPBC reads/writes BP_CONST.TXT etc. in the run folder, not e:\temp\
    "MINOR",         # #18 explicit exit status 0 (Python already has no argv[0] path or for(i=i) issue)
)
ALL_FIXES = C_FIXES + PY_FIXES
FIX = {t.strip().upper() for t in os.environ.get("EQ_FIX", "").split(",") if t.strip()}
if not FIX or "ALL" in FIX:
    FIX = set(ALL_FIXES)
FIX -= {t.strip().upper() for t in os.environ.get("EQ_NOFIX", "").split(",") if t.strip()}  # EQ_NOFIX=TAG,.. drops some
_bad = FIX - set(ALL_FIXES)
if _bad:
    raise SystemExit("unknown EQ_FIX tag(s): %s" % ",".join(sorted(_bad)))
MAX_ATTEMPTS = int(os.environ.get("EQ_MAX_ATTEMPTS", "10"))


def fixed(tag):
    return tag in FIX


def trial_seed():
    """srand() call made before every trial/retry in 1995 (from the clock)."""
    if not fixed("SEED-ONCE"):
        srand(int(time.time()) & 0xFFFF)


def program_seed():
    """SEED-ONCE: one srand() at start; EQ_SEED (k=0) or 1."""
    if fixed("SEED-ONCE"):
        srand(1)


def preamble_bit(pn_shift):
    """Next training symbol: the 5-bit PN (31-bit period) as in 1995, or a random bit."""
    if fixed("RAND-PREAMBLE"):
        return float(sign((2.0 * URAND()) - 1.0))
    return next_pn(pn_shift)


def finish():
    if fixed("MINOR"):
        sys.exit(0)


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

    program_seed()
    passed = gaveup = 0
    errors_all = [0.0] * bits
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        fail = 1
        attempts = 0
        while fail:
            trial_seed()
            attempts += 1

            for i in range(m_taps):
                modelline[i].value = -1.0
            for i in range(ann.nodes_0):
                ann.layer_0[i].output = -1.0
            if fixed("LMS-BIAS"):
                ann.layer_0[ann.nodes_0].output = -1.0

            if fixed("LMS-FILL"):
                fill_rand(ann.layer_1, ann.nodes_0 + 1, ann.nodes_1)
            else:
                fill_rand(ann.layer_1, ann.nodes_0 + ann.nodes_1 + 1, ann.nodes_1)

            signal = 1.0 if not fixed("RAND-PREAMBLE") else preamble_bit(pn_shift)
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

                signal = preamble_bit(pn_shift)
                shift_right_layer(ann.layer_0, ann.nodes_0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)

            if (mean_err / 400) < 0.6:
                fail = 0
                sys.stderr.write(" %1.3f PASSED" % (mean_err / 400))
            else:
                sys.stderr.write(" %1.3f FAILED" % (mean_err / 400))
                failiures += 1
                if fixed("RETRY-CAP") and attempts >= MAX_ATTEMPTS:
                    break

        if fixed("RETRY-CAP"):   # every trial's last attempt, passed or not
            for i in range(bits - delay):
                errors_all[i] += error_run[i]
        if not fail:
            passed += 1
            for i in range(bits - delay):
                errors[i] += error_run[i]
        else:
            gaveup += 1

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

    if fixed("RETRY-CAP"):   # curve = mean over the trials that passed; BER over all trials
        save_doubles(errors if passed else [float("nan")] * bits, bits - delay, max(passed, 1),
                     path, "ERRORS.TXT")
        save_doubles([gaveup], 1, runs, path, "GAVEUP.TXT")
        save_doubles(errors_all, bits - delay, runs, path, "ERRORS_ALL.TXT")
    else:
        save_doubles(errors, bits - delay, runs, path, "ERRORS.TXT")
    save_doubles([ber], 1, runs, path, "BER.TXT")
    save_doubles([failiures], 1, runs, path, "FAIL.TXT")
    finish()


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

    n_in = n0 + n1 + 1 if fixed("RNN-FFWD") else n0 + 1   # inputs fed to the nodes
    kron_div = fixed("RNN-KRON")
    program_seed()
    passed = gaveup = 0
    errors_all = [0.0] * bits
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        fail = 1
        attempts = 0
        while fail:
            trial_seed()
            attempts += 1
            for i in range(m_taps):
                modelline[i].value = -1.0
            for k in range(n1):
                for l in range(n0 + n1 + 1):
                    for j in range(n1):
                        L1[k].weights[l].sensitivity[j].sense = 0.0
            fill_rand(L1, n0 + n1 + 1, n1)
            L0[n0].output = -1.0
            signal = 1.0 if not fixed("RAND-PREAMBLE") else preamble_bit(pn_shift)
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
                                if kron_div:   # FIX RNN-KRON: /nodes_1 inside the sum, as in C
                                    for i in range(n0 + 1, n0 + 1 + n1):
                                        s.sum += (L1[j].weights[i].weight
                                                  * L1[k].weights[l].sensitivity[i - n0 - 1].sense
                                                  + KRONECKER(k, j) * L0[l].output / n1)
                                else:
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
                signal = preamble_bit(pn_shift)
                shift_right_layer(L0, n0)
                shift_right_fir(modelline, m_taps)
                shift_right_fir(delayline, delay + 1)
            if (mean_err / 400) < 0.6:
                fail = 0
            else:
                failiures += 1
                if fixed("RETRY-CAP") and attempts >= MAX_ATTEMPTS:
                    break
        if fixed("RETRY-CAP"):   # every trial's last attempt, passed or not
            for i in range(bits - delay):
                errors_all[i] += error_run[i]
        if not fail:
            passed += 1
            for i in range(bits - delay):
                errors[i] += error_run[i]
        else:
            gaveup += 1
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
    if fixed("RETRY-CAP"):
        save_doubles(errors if passed else [float("nan")] * bits, bits - delay, max(passed, 1),
                     path, "ERRORS.TXT")
        save_doubles([gaveup], 1, runs, path, "GAVEUP.TXT")
        save_doubles(errors_all, bits - delay, runs, path, "ERRORS_ALL.TXT")
    else:
        save_doubles(errors, bits - delay, runs, path, "ERRORS.TXT")
    save_doubles([ber], 1, runs, path, "BER.TXT")
    save_doubles([failiures], 1, runs, path, "FAIL.TXT")
    finish()


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
    program_seed()
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        trial_seed()
        for i in range(ann.nodes_1):
            for j in range(ann.nodes_0):
                ann.layer_1[i].weights[j].weight = 0.0
        for i in range(m_taps):
            modelline[i].value = -1.0
        for i in range(ann.nodes_0):
            ann.layer_0[i].output = -1.0
        for i in range(ann.nodes_1):
            ann.layer_1[i].counter = 0
        signal = 1.0 if not fixed("RAND-PREAMBLE") else preamble_bit(pn_shift)
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
            signal = preamble_bit(pn_shift)
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
    finish()


# ---------------------------------------------------------------------------
# BPBC
def fname(dos_name, plain):
    return plain if fixed("BP-PATHS") else dos_name


def bpbc_main(no_pars, par_txt):
    with open(fname("e:\\temp\\bp_const.txt", "BP_CONST.TXT")) as f:
        v = f.read().split()
    m_taps, bits, epochs, snr, runs = int(v[0]), int(v[1]), int(v[2]), float(v[3]), int(v[4])
    ann.nodes_0, ann.nodes_1, ann.nodes_2 = int(v[5]), int(v[6]), int(v[7])
    ann.learn_rate, ann.momentum = float(v[8]), float(v[9])
    ann.nodes_3 = 1
    modelline = calloc(m_taps, FIR_MODEL_STR)
    with open(fname("e:\\temp\\weights.txt", "WEIGHTS.TXT")) as f:
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
    program_seed()
    for run_num in range(runs):
        sys.stderr.write("TRIAL:%03d" % (run_num + 1))
        trial_seed()
        for i in range(m_taps):
            modelline[i].value = -1.0
        signal = 1.0 if not fixed("RAND-PREAMBLE") else preamble_bit(pn_shift)
        i = 0
        while i < (bits + math.ceil(n0 / 2.0)):
            train_vect[i][1] = modelline[0].value = signal
            train_vect[i][0] = fir_output(modelline, m_taps) + noise(snr)
            signal = preamble_bit(pn_shift)
            shift_right_fir(modelline, m_taps)
            i += 1

        if STRICT and not fixed("BP-DEBUG"):
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

    with open(fname("e:\\temp\\errors.txt", "ERRORS.TXT"), "w") as f:
        for i in range(epochs):
            f.write("%+2.15f\n" % (errors[i] / runs))
    with open(fname("e:\\temp\\ber.txt", "BER.TXT"), "w") as f:
        f.write("%+2.15f\n" % (ber / runs))
    finish()


PROGRAMS = {"LMS": lms_main, "RNN": rnn_main, "RB": rb_main, "BPBC": bpbc_main}


def cli(programs):
    if len(sys.argv) < 2 or sys.argv[1].upper() not in programs:
        sys.exit("usage: %s {%s} [TEST]" % (os.path.basename(sys.argv[0]), ",".join(programs)))
    argv = sys.argv[1:]          # argv[1] is TEST, as in the DOS program's own command line
    programs[argv[0].upper()](len(argv), argv)


if __name__ == "__main__":
    cli(PROGRAMS)
