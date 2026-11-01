// Verilator testbench for bp_engine: tb JOB.bin OUT.bin [trials] -> trains each trial, writes weights, prints cycles.
#include "Vbp_engine.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

static Vbp_engine *m;
static uint64_t cyc;
static void tick() { m->clk = 0; m->eval(); m->clk = 1; m->eval(); cyc++; }

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    FILE *f = fopen(argv[1], "rb");
    int32_t h[5];
    if (fread(h, 4, 5, f) != 5) return 1;
    int B = h[0], E = h[1], K1 = h[2], K2 = h[3], T = h[4];
    if (argc > 3) T = atoi(argv[3]) < T ? atoi(argv[3]) : T;
    m = new Vbp_engine;
    FILE *o = fopen(argv[2], "wb");
    uint64_t total = 0;
    for (int t = 0; t < T; t++) {
        std::vector<int32_t> xm(B + 1), w(67);
        if (fread(xm.data(), 4, B + 1, f) != (size_t)(B + 1) || fread(w.data(), 4, 67, f) != 67) return 1;
        m->nsamp = B; m->epochs = E; m->k1 = K1; m->k2 = K2; m->start = 0;
        for (int i = 0; i <= B; i++) { m->h_we = 1; m->h_mem = 1; m->h_addr = i; m->h_wdata = xm[i]; tick(); }
        for (int i = 0; i < 67; i++) { m->h_we = 1; m->h_mem = 0; m->h_addr = i; m->h_wdata = w[i]; tick(); }
        m->h_we = 0; m->start = 1; tick(); m->start = 0;
        uint64_t c0 = cyc;
        while (m->busy) tick();
        total += cyc - c0;
        for (int i = 0; i < 67; i++) { m->h_addr = i; tick(); tick(); w[i] = (int32_t)m->h_rdata; }
        fwrite(w.data(), 4, 67, o);
    }
    fclose(o);
    printf("trials=%d cycles/trial=%.0f cycles/presentation=%.1f\n", T, (double)total / T, (double)total / T / E / B);
    delete m;
    return 0;
}
