#!/bin/sh
# Build src/clang on another host from already-prepared sources (no uv/Python needed there).
# Layout (as synced by scenarios/run_scenarios.py with REMOTE=host): build/ build_fix/ compat/ openwatcom/
set -e
DC=${DC:-openwatcom}   # ../openwatcom when run inside the repo
CF="-O2 -std=gnu89 -w -Dfar= -Icompat -I$DC"
for B in build build_fix*; do
  for p in lms rnn rb bpbc; do
    init=${p}_init.h; [ $p = bpbc ] && init=bp_init.h
    [ -f $B/$p.c ] || continue
    extra=""; [ $p = rb ] && extra="-include $DC/rb_missing.h"
    clang $CF $extra -I$B -o $B/$p $B/$p.c -x c $B/$init -x none $B/eq_funcs.c $DC/eqcompat.c -lm
  done
done
file build/lms | grep -q arm64 && echo "built (arm64)"
