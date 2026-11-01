#!/bin/sh
# Enumerate memory/UB faults in the 1995 programs: build with ASan+UBSan (no heap guard),
# run short versions of the report scenarios, print each distinct fault once.
# Usage: src/clang/sanitize.sh [build|build_fix]   (from the repo root, after make -C src/clang all fixed)
B=${1:-build}
W=$(mktemp -d); P=$(pwd)/src/clang
CF="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fsanitize-recover=address -std=gnu89 -w -Dfar= -DNO_HEAP_GUARD -I$P/compat -I$P/../openwatcom -I$P/$B"
for p in lms rnn rb bpbc; do
  init=${p}_init.h; [ $p = bpbc ] && init=bp_init.h
  extra=""; [ $p = rb ] && extra="-include $P/../openwatcom/rb_missing.h"
  clang $CF $extra -o $W/$(echo $p | tr a-z A-Z).EXE $P/$B/$p.c -x c $P/$B/$init -x none $P/$B/eq_funcs.c $P/../openwatcom/eqcompat.c -lm
done
cd $W
run() {  # prog con weights
  printf "$2" > $1_CON.TXT; printf "$3" > WEIGHTS.TXT
  printf "$2" > 'e:\temp\bp_const.txt'; printf "$3" > 'e:\temp\weights.txt'
  echo "== $1: $2" | tr -d '\n'; echo
  EQ_SEED=1 ASAN_OPTIONS=halt_on_error=0 UBSAN_OPTIONS=print_stacktrace=1 \
    perl -e 'alarm 60; exec @ARGV' ./$1.EXE TEST 2>&1 >/dev/null | LC_ALL=C grep -aE "ERROR: AddressSanitizer|runtime error|SUMMARY|^    #[0-2] .* in [a-z_]+ " | \
    LC_ALL=C sed -E 's/0x[0-9a-f]+//g; s#[^ ]*src/clang/#src/clang/#g; s/[^ -~]+//g; s/==[0-9]+==//; s/\(.*\)//' | sort | uniq -c | sort -rn | head -12
}
run LMS "2 1000 21 3 0 7 0.01\n" "1.0\n0.5\n"
run LMS "10 1000 20 3 5 15 0.008\n" "0.047\n0.321\n0.783\n0.427\n0.293\n0.100\n0.041\n0.038\n0.020\n0.011\n"
run RNN "2 1000 21 3 0 2 3 0.005\n" "1.0\n0.5\n"
run RNN "10 1000 20 3 5 10 4 0.003\n" "0.047\n0.321\n0.783\n0.427\n0.293\n0.100\n0.041\n0.038\n0.020\n0.011\n"
run RB "2 1000 10 3 2 8 0.125\n" "1.0\n0.5\n"
run BPBC "2 40 100 21 2 2 6 6 0.1 0.5\n" "1.0\n0.5\n"
rm -rf $W
