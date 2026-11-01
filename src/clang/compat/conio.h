/* Minimal Borland <conio.h> for a hosted (macOS/Linux) build of the 1995 sources.
   Screen-positioning and colour calls are no-ops; cprintf() (only used for the
   start-up banner) goes to stderr so stdout stays clean. getch()/kbhit() never
   block: kbhit() returns 1 (a key is always waiting) and getch() returns a space,
   because BPBC.C as printed has two unguarded "wait for a key" debug pauses. */
#ifndef PORT_CONIO_H
#define PORT_CONIO_H
#include <stdio.h>
#include <stdarg.h>
#define BLACK 0
#define BLUE 1
#define WHITE 15
#define _NOCURSOR 0
#define _NORMALCURSOR 2
static void gotoxy(int x, int y) { (void)x; (void)y; }
static int  wherey(void) { return 4; }
static void clrscr(void) {}
static void textcolor(int c) { (void)c; }
static void textbackground(int c) { (void)c; }
static void highvideo(void) {}
static void lowvideo(void) {}
static void _setcursortype(int t) { (void)t; }
static int  kbhit(void) { return 1; }   /* "a key is waiting": BPBC.C as printed pauses for a key twice per trial */
static int  getch(void) { return ' '; }
static int  putch(int c) { return fputc(c, stderr); }
static int  cprintf(const char *f, ...) { int n; va_list a; va_start(a, f); n = vfprintf(stderr, f, a); va_end(a); return n; }
/* Heap guard: every calloc gets one spare element. LMS.C's fill_rand writes one
   weight past its array (see results/RESULTS.md); on macOS that write either
   lands in the next block or corrupts allocator metadata and traps at random.
   The spare element absorbs it, so the faithful build runs reliably. The DOS
   build has no guard and shows the corruption (first point of ERRORS.TXT). */
#include <stdlib.h>
#ifndef NO_HEAP_GUARD   /* -DNO_HEAP_GUARD for the sanitizer builds */
static void *eq_calloc(size_t n, size_t s) { return calloc(n + 1, s); }
#define calloc(n, s) eq_calloc(n, s)
#endif
#include "eqcompat.h"
#endif
