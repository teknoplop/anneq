/* bpfix.c - bit-exact fixed-point reference of the FPGA backprop engine (gateware/rtl/bp_engine.v),
 * plus a double-precision BPBC.C-equivalent trainer for the native speed baseline.
 *
 * Network: BPBC 2-6-6-1, tanh units, bias = extra input fixed at -1, batch training with momentum.
 * Fixed-point formats (all shifts are arithmetic, i.e. floor):
 *   activations, inputs, deltas : 18-bit Q3.14        weights : 32-bit Q3.28 master, used as w>>14 (Q3.14)
 *   products                   : Q6.28              gradient sums : 36-bit Q.28
 *   tanh                       : LUT[min(|s|>>5, 2047)], s = sum>>14, LUT[i] = round(16384*tanh((i+0.5)/512))
 *   epoch update               : dl = (g*K1 + dl*K2) >> 16 ; w += dl   (K1 = lr(1-mom)*65536, K2 = mom*65536)
 * Weight index map (67): w1[n][j] = n*3+j (n<6, j<3), w2[n][j] = 18+n*7+j (j<7), w3[j] = 60+j.
 *
 * Usage:
 *   bpfix fix  LUT.hex JOB.bin OUT.bin     JOB = int32 B, epochs, K1, K2, T; then per trial int32 xm[B+1], int32 w[67]
 *                                          OUT = per trial int32 w[67]
 *   bpfix bench LUT.hex JOB.bin            times fixed and double training of all trials (one thread)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NW 67
static int32_t LUT[2048];

static int32_t sat18(int64_t v) { return v > 131071 ? 131071 : v < -131072 ? -131072 : (int32_t)v; }
static int32_t w18(int32_t w) { return w >> 14; }
static int32_t tanh_q(int64_t sum)
{
    int64_t s = sum >> 14, m = s < 0 ? -s : s, idx = m >> 5;
    if (idx > 2047) idx = 2047;
    return s < 0 ? -LUT[idx] : LUT[idx];
}
static int32_t der(int32_t h) { return 16384 - (int32_t)(((int64_t)h * h) >> 14); }
static int32_t bk(int64_t sum, int32_t h) { return sat18(((int64_t)der(h) * sat18(sum >> 14)) >> 14); }

/* xm[i]: bits 15..0 = input sample Q2.14 (signed), bit 16 = desired (+1 if set) */
void train_fix(const int32_t *xm, int B, int32_t *w, int epochs, int32_t K1, int32_t K2)
{
    int64_t g[NW];
    int32_t dl[NW] = {0};
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < NW; i++) g[i] = 0;
        for (int s = 0; s < B; s++) {
            int32_t a0[3], a1[7], a2[7], y, d1[6], d2[6], d3;
            a0[0] = (int16_t)(xm[s + 1] & 0xffff);
            a0[1] = (int16_t)(xm[s] & 0xffff);
            a0[2] = a1[6] = a2[6] = -16384;
            int32_t D = (xm[s + 1] >> 16) & 1 ? 16384 : -16384;
            for (int n = 0; n < 6; n++) { int64_t acc = 0; for (int j = 0; j < 3; j++) acc += (int64_t)w18(w[n*3+j]) * a0[j]; a1[n] = tanh_q(acc); }
            for (int n = 0; n < 6; n++) { int64_t acc = 0; for (int j = 0; j < 7; j++) acc += (int64_t)w18(w[18+n*7+j]) * a1[j]; a2[n] = tanh_q(acc); }
            { int64_t acc = 0; for (int j = 0; j < 7; j++) acc += (int64_t)w18(w[60+j]) * a2[j]; y = tanh_q(acc); }
            d3 = bk((int64_t)(D - y) * 16384, y);
            for (int n = 0; n < 6; n++) d2[n] = bk((int64_t)w18(w[60+n]) * d3, a2[n]);
            for (int n = 0; n < 6; n++) { int64_t acc = 0; for (int k = 0; k < 6; k++) acc += (int64_t)w18(w[18+k*7+n]) * d2[k]; d1[n] = bk(acc, a1[n]); }
            for (int j = 0; j < 7; j++) g[60+j] += (int64_t)d3 * a2[j];
            for (int n = 0; n < 6; n++) for (int j = 0; j < 7; j++) g[18+n*7+j] += (int64_t)d2[n] * a1[j];
            for (int n = 0; n < 6; n++) for (int j = 0; j < 3; j++) g[n*3+j] += (int64_t)d1[n] * a0[j];
        }
        for (int i = 0; i < NW; i++) {
            int64_t gg = (int64_t)((uint64_t)g[i] << 28) >> 28;   /* 36-bit register in hardware */
            int64_t t = gg * K1 + (int64_t)dl[i] * K2;
            dl[i] = (int32_t)(t >> 16);
            w[i] = (int32_t)((uint32_t)w[i] + (uint32_t)dl[i]);
        }
    }
}

/* Same algorithm in double precision, as BPBC.C does it (weights fixed during an epoch). */
void train_dbl(const double *x, const double *d, int B, double *w, int epochs, double lr, double mom)
{
    double g[NW], dl[NW] = {0};
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < NW; i++) g[i] = 0;
        for (int s = 0; s < B; s++) {
            double a0[3] = {x[s+1], x[s], -1}, a1[7], a2[7], y, d1[6], d2[6], d3;
            a1[6] = a2[6] = -1;
            for (int n = 0; n < 6; n++) { double acc = 0; for (int j = 0; j < 3; j++) acc += w[n*3+j] * a0[j]; a1[n] = tanh(acc); }
            for (int n = 0; n < 6; n++) { double acc = 0; for (int j = 0; j < 7; j++) acc += w[18+n*7+j] * a1[j]; a2[n] = tanh(acc); }
            { double acc = 0; for (int j = 0; j < 7; j++) acc += w[60+j] * a2[j]; y = tanh(acc); }
            d3 = (1 - y*y) * (d[s+1] - y);
            for (int n = 0; n < 6; n++) d2[n] = w[60+n] * d3 * (1 - a2[n]*a2[n]);
            for (int n = 0; n < 6; n++) { double acc = 0; for (int k = 0; k < 6; k++) acc += w[18+k*7+n] * d2[k]; d1[n] = acc * (1 - a1[n]*a1[n]); }
            for (int j = 0; j < 7; j++) g[60+j] += d3 * a2[j];
            for (int n = 0; n < 6; n++) for (int j = 0; j < 7; j++) g[18+n*7+j] += d2[n] * a1[j];
            for (int n = 0; n < 6; n++) for (int j = 0; j < 3; j++) g[n*3+j] += d1[n] * a0[j];
        }
        for (int i = 0; i < NW; i++) { dl[i] = g[i] * lr * (1 - mom) + mom * dl[i]; w[i] += dl[i]; }
    }
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: bpfix fix|bench LUT.hex JOB.bin [OUT.bin]\n"); return 2; }
    FILE *f = fopen(argv[2], "r");
    if (!f) { perror(argv[2]); return 1; }
    for (int i = 0; i < 2048; i++) { unsigned v; if (fscanf(f, "%x", &v) != 1) return 1; LUT[i] = (int32_t)v; }
    fclose(f);
    f = fopen(argv[3], "rb");
    if (!f) { perror(argv[3]); return 1; }
    int32_t hdr[5];
    if (fread(hdr, 4, 5, f) != 5) return 1;
    int B = hdr[0], E = hdr[1], K1 = hdr[2], K2 = hdr[3], T = hdr[4];
    int32_t *xm = malloc(sizeof(int32_t) * T * (B + 1)), *w = malloc(sizeof(int32_t) * T * NW);
    for (int t = 0; t < T; t++) {
        if (fread(xm + t*(B+1), 4, B + 1, f) != (size_t)(B + 1) || fread(w + t*NW, 4, NW, f) != NW) return 1;
    }
    fclose(f);
    if (argv[1][0] == 'f') {
        for (int t = 0; t < T; t++) train_fix(xm + t*(B+1), B, w + t*NW, E, K1, K2);
        f = fopen(argv[4], "wb");
        fwrite(w, 4, (size_t)T * NW, f);
        fclose(f);
        return 0;
    }
    /* bench: double then fixed, all T trials sequentially, report pattern presentations/s */
    double *xd = malloc(sizeof(double) * T * (B + 1)), *dd = malloc(sizeof(double) * T * (B + 1)), *wd = malloc(sizeof(double) * T * NW);
    for (int i = 0; i < T * (B + 1); i++) { xd[i] = (int16_t)(xm[i] & 0xffff) / 16384.0; dd[i] = (xm[i] >> 16) & 1 ? 1 : -1; }
    for (int i = 0; i < T * NW; i++) wd[i] = w[i] / 268435456.0;
    double lr = 0.1, mom = 0.5, t0 = now(), chk = 0;
    for (int t = 0; t < T; t++) train_dbl(xd + t*(B+1), dd + t*(B+1), B, wd + t*NW, E, lr, mom);
    double t1 = now();
    for (int t = 0; t < T; t++) train_fix(xm + t*(B+1), B, w + t*NW, E, K1, K2);
    double t2 = now();
    for (int i = 0; i < T * NW; i++) chk += wd[i] + w[i] / 268435456.0;
    double n = (double)T * E * B;
    printf("trials=%d epochs=%d samples=%d presentations=%.0f\n", T, E, B, n);
    printf("double: %.4f s  %.3e presentations/s  %.3e epoch-updates/s\n", t1 - t0, n / (t1 - t0), (double)T * E / (t1 - t0));
    printf("fixed : %.4f s  %.3e presentations/s  %.3e epoch-updates/s\n", t2 - t1, n / (t2 - t1), (double)T * E / (t2 - t1));
    printf("checksum %g\n", chk);
    return 0;
}
