"""Make bug-fixed copies of the prepared 1995 sources (for with/without-fix runs).

Usage: apply_fixes.py PREP_DIR OUT_DIR [TAG,TAG..]   (PREP_DIR = output of prepare.py;
       default: all fixes)
Each fix is a textual substitution that must match exactly once:
  LMS-FILL   lms.c: fill_rand() is told nodes_0+nodes_1+1 = nodes_0+2 weights but the
             array holds nodes_0+1, so it writes one weight past the end (heap
             overrun; copied from RNN.C where the count is right). Use nodes_0+1.
  LMS-BIAS   lms.c: the ADALINE bias input layer_0[nodes_0] is never set (stays 0
             from calloc), so the network has no bias. Set it to -1.0 as the
             report's Figure 5.1 shows.
  RNN-FFWD   rnn.c: ffwd_sig() is given nodes_0+1 inputs, so the fed-back outputs
             (inputs nodes_0+1 .. nodes_0+nodes_1) never reach the nodes: the
             "recurrent" net runs as a plain single-layer tanh perceptron.
             Feed all nodes_0+nodes_1+1 inputs (training and test).
  RNN-KRON   rnn.c: the RTRL term delta_kj*u_l sits inside the sum over the
             fed-back nodes, so it is added nodes_1 times. Divide it by nodes_1
             (same as adding it once).
  BP-DEBUG   bpbc.c: two weight dumps + "press a key" pauses per trial are not inside
             #ifdef DEBUG, and they call print_weights() with its layer arguments
             swapped, reading the input layer's (NULL) weight pointers. Harmless on
             real-mode DOS (reads low memory), a segfault on a hosted OS. Removed.
"""
import pathlib, re, shutil, sys

FIXES = {
    "lms.c": [
        ("LMS-FILL", r"fill_rand\(ann\.layer_1, ann\.nodes_0\+ann\.nodes_1\+1, ann\.nodes_1\);",
         "fill_rand(ann.layer_1, ann.nodes_0+1, ann.nodes_1); /* FIX LMS-FILL */", 1),
        ("LMS-BIAS", r"(fill_rand\(ann\.layer_1, ann\.nodes_0\+[^;]*;)",
         "ann.layer_0[ann.nodes_0].output = -1.0; /* FIX LMS-BIAS */ \\1", 1),
    ],
    "rnn.c": [
        ("RNN-FFWD", r"ffwd_sig\(ann\.layer_0, ann\.layer_1, ann\.nodes_0\+1,\s*ann\.nodes_1\);",
         "ffwd_sig(ann.layer_0, ann.layer_1, ann.nodes_0+ann.nodes_1+1, ann.nodes_1); /* FIX RNN-FFWD */", 2),
        ("RNN-KRON", r"KRONECKER\(k,j\)\*ann\.layer_0\[l\]\.output\)",
         "KRONECKER(k,j)*ann.layer_0[l].output/ann.nodes_1) /* FIX RNN-KRON */", 1),
    ],
    "bpbc.c": [
        ("BP-DEBUG", r"gotoxy\(1, 1\);\s*print_weights\([^;]*;\s*print_weights\([^;]*;\s*print_weights\([^;]*;\s*printf\(\"\\n\"\);\s*while\(!kbhit\(\)\);\s*getch\(\);",
         "/* FIX BP-DEBUG: debug weight dump + key wait removed */", 3),
    ],
}

src, dst = map(pathlib.Path, sys.argv[1:3])
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
for name, fixes in FIXES.items():
    p = dst / name
    t = p.read_text()
    for tag, pat, rep, n in fixes:
        if only and tag not in only:
            continue
        t, k = re.subn(pat, rep, t)
        if k != n:
            sys.exit(f"{tag}: expected {n} match(es) in {name}, got {k}")
        print(f"{tag}: {k} substitution(s) in {name}")
    p.write_text(t)
