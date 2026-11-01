/* See eqcompat.h. Compiled without the macros so it can call the real libc. */
#include <stdlib.h>
#undef rand    /* in case a force-included shim defined them */
#undef srand
static unsigned long bc_seed = 1;
static int use_libc = -1;
static long seed_base = -1, seed_k = 0;

static void init(void) {
    char *e;
    if (use_libc >= 0) return;
    e = getenv("EQ_RAND");
    use_libc = (e != NULL && (e[0] == 'l' || e[0] == 'L'));
    e = getenv("EQ_SEED");
    seed_base = e ? atol(e) : -1;
}
int eq_rand(void) {
    init();
    if (use_libc) return (int)((double)rand() / RAND_MAX * 0x7FFF);
    bc_seed = (bc_seed * 22695477UL + 1UL) & 0xFFFFFFFFUL;
    return (int)((bc_seed >> 16) & 0x7FFF);
}
void eq_srand(unsigned s) {
    init();
    if (seed_base >= 0) s = (unsigned)(seed_base + seed_k++);
    bc_seed = s;
    srand(s);
}
