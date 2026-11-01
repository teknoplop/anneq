/* Shared shim for every rebuild of the 1995 sources (DOS/Watcom and clang).
   Included after <stdlib.h>, it replaces rand()/srand() with:
   - Borland C++ 3.1's generator (seed = seed*22695477 + 1; return (seed>>16)&0x7FFF,
     RAND_MAX 0x7FFF), so every build draws the same numbers the 1995 .EXEs did;
   - optional deterministic seeding: if the environment variable EQ_SEED is set,
     the k-th srand() call (k = 0,1,2..) uses EQ_SEED + k instead of time().
     Without it the programs seed from time() exactly as in 1995, which on a
     fast host gives every trial in the same second the same random numbers.
   - EQ_RAND=libc keeps the host C library rand() instead (for comparison). */
#ifndef EQCOMPAT_H
#define EQCOMPAT_H
#include <stdlib.h>
int  eq_rand(void);
void eq_srand(unsigned s);
#undef RAND_MAX
#define RAND_MAX 0x7FFF
#define rand() eq_rand()
#define srand(s) eq_srand(s)
#endif
