# C++23 port of the four 1995 equaliser simulators (issue #25).
#   make -C src/clang/anneq            release build: build/{lms,rnn,rb,bpbc}
#   make -C src/clang/anneq debug      libc++ hardening (bounds-checked vectors): build_debug/
#   make -C src/clang/anneq sanitize   ASan + UBSan: build_san/
#   make -C src/clang/anneq check      byte-for-byte check (release, debug and sanitize builds) against the C builds and anneq_fixed.py
# -ffp-contract=off: no fused multiply-add, so the arithmetic rounds exactly as the C/Python
# reference (src/tools/check_pyport.py compiles its C references the same way).
CXX      ?= clang++
STD      := -std=c++23 -Wall -Wextra -Wpedantic -ffp-contract=off
CXXFLAGS ?= -O2
PROGS    := lms rnn rb bpbc
HARDEN   := -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG -D_GLIBCXX_ASSERTIONS
SAN      := -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer

all: $(PROGS:%=build/%)
debug: $(PROGS:%=build_debug/%)
sanitize: $(PROGS:%=build_san/%)

build/%: %.cpp eq_funcs.hpp ; @mkdir -p $(@D) && $(CXX) $(STD) $(CXXFLAGS) -o $@ $<
build_debug/%: %.cpp eq_funcs.hpp ; @mkdir -p $(@D) && $(CXX) $(STD) -O1 -g $(HARDEN) -o $@ $<
build_san/%: %.cpp eq_funcs.hpp ; @mkdir -p $(@D) && $(CXX) $(STD) -O1 -g $(HARDEN) $(SAN) -o $@ $<

check: all debug sanitize
	$(MAKE) -C .. all fixed variants
	for b in build build_debug build_san; do \
	  (cd ../../.. && EQ_CPP_BUILD=$$b uv run --no-project python src/tools/check_cpp.py) || exit 1; done

clean: ; rm -rf build build_debug build_san
.PHONY: all debug sanitize check clean
