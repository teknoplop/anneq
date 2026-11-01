#!/bin/sh
# Build the four 1995 programs as 16-bit DOS large-model .EXEs with Open Watcom 2.0.
# Usage: src/openwatcom/build.sh   (from the repo root; output in build/dos/)
set -e
: "${WATCOM:=$HOME/opt/watcom}"
export WATCOM INCLUDE="$WATCOM/h" PATH="$WATCOM/bino64:$PATH"
uv run --no-project python src/tools/prepare.py src/borland build/dos
cd build/dos
for f in lms_init rnn_init rb_init bp_init; do cp $f.h $f.c; done
W="wcl -q -bt=dos -ml -fpi87 -fi=../../src/openwatcom/borland.h -i=../../src/openwatcom"
wcl -q -bt=dos -ml -fpi87 -c ../../src/openwatcom/eqcompat.c -fo=eqcompat.obj
$W -c eq_funcs.c -fo=eq_funcs.obj
$W lms.c  lms_init.c eq_funcs.obj eqcompat.obj -fe=LMS.EXE
$W rnn.c  rnn_init.c eq_funcs.obj eqcompat.obj -fe=RNN.EXE
$W -dRB_MISSING rb.c rb_init.c eq_funcs.obj eqcompat.obj -fe=RB.EXE
$W bpbc.c bp_init.c eq_funcs.obj eqcompat.obj -fe=BPBC.EXE
ls -l *.EXE
