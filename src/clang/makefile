# clang build of the unmodified 1995 sources (src/borland/) on macOS/Linux.
# The listings are first un-wrapped by src/tools/prepare.py (print line-wrap only).
# Changes vs Borland C++ 3.1, all outside src/borland/:
#   -Dfar=        16-bit far pointers do not exist on a flat host
#   compat/conio.h  no-op screen calls (see file); ../openwatcom/eqcompat.[ch]
#                   Borland rand(), optional EQ_SEED deterministic seeding
CC      ?= clang
CFLAGS  ?= -O2 -std=gnu89 -w -Dfar= -Icompat -I../openwatcom
B       := build
PROGS   := $(B)/lms $(B)/rnn $(B)/rb $(B)/bpbc

all: $(PROGS)

$(B)/.prep: $(wildcard ../borland/*)
	cd ../.. && uv run --no-project python src/tools/prepare.py src/borland src/clang/$(B)
	touch $@

$(B)/lms:  $(B)/.prep ; $(CC) $(CFLAGS) -I$(B) -o $@ $(B)/lms.c  -x c $(B)/lms_init.h -x none $(B)/eq_funcs.c ../openwatcom/eqcompat.c -lm
$(B)/rnn:  $(B)/.prep ; $(CC) $(CFLAGS) -I$(B) -o $@ $(B)/rnn.c  -x c $(B)/rnn_init.h -x none $(B)/eq_funcs.c ../openwatcom/eqcompat.c -lm
$(B)/rb:   $(B)/.prep ; $(CC) $(CFLAGS) -include ../openwatcom/rb_missing.h -I$(B) -o $@ $(B)/rb.c   -x c $(B)/rb_init.h -x none $(B)/eq_funcs.c ../openwatcom/eqcompat.c -lm
$(B)/bpbc: $(B)/.prep ; $(CC) $(CFLAGS) -I$(B) -o $@ $(B)/bpbc.c -x c $(B)/bp_init.h -x none $(B)/eq_funcs.c ../openwatcom/eqcompat.c -lm

clean: ; rm -rf $(B) $(F) $(F)_*
.PHONY: all clean

# Bug-fixed variants (src/fixes/*.patch, one per fix, each documented in its header): build_fix/lms, build_fix/rnn, build_fix/bpbc
# (build/bpbc is the faithful build; it segfaults on a hosted OS, see BP-DEBUG)
F := build_fix
fixed: $(B)/.prep
	rm -rf $(F) && cp -R $(B) $(F) && rm -f $(F)/lms $(F)/rnn $(F)/rb $(F)/bpbc
	for p in ../fixes/*.patch; do patch -s -p1 -d $(F) < $$p; done
	$(CC) $(CFLAGS) -I$(F) -o $(F)/lms $(F)/lms.c -x c $(F)/lms_init.h -x none $(F)/eq_funcs.c ../openwatcom/eqcompat.c -lm
	$(CC) $(CFLAGS) -I$(F) -o $(F)/rnn $(F)/rnn.c -x c $(F)/rnn_init.h -x none $(F)/eq_funcs.c ../openwatcom/eqcompat.c -lm
	$(CC) $(CFLAGS) -I$(F) -o $(F)/bpbc $(F)/bpbc.c -x c $(F)/bp_init.h -x none $(F)/eq_funcs.c ../openwatcom/eqcompat.c -lm
.PHONY: fixed

# One-fix-at-a-time variants: build_fix_<TAG>/ (sources only; built by remote_build.sh or `make variants`)
TAGS := LMS-FILL LMS-BIAS RNN-FFWD RNN-KRON
variants: $(B)/.prep
	for t in $(TAGS); do rm -rf $(F)_$$t && cp -R $(B) $(F)_$$t && rm -f $(F)_$$t/lms $(F)_$$t/rnn $(F)_$$t/rb $(F)_$$t/bpbc && patch -s -p1 -d $(F)_$$t < ../fixes/$$t.patch; done
	DC=../openwatcom sh remote_build.sh
.PHONY: variants

# The C++20 port (src/clang/anneq/, issue #25) builds separately; this target is a shortcut.
cpp: ; $(MAKE) -C anneq
.PHONY: cpp
