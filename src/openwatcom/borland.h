/* Borland C++ 3.1 conio extras for Open Watcom (16-bit DOS).
   Force-included with wcl -fi=borland.h so src/borland/ stays untouched.
   Each function maps to a BIOS int 10h call; colours are ignored
   (cprintf in Watcom writes plain text), which only affects the banner. */
#ifndef BORLAND_COMPAT_H
#define BORLAND_COMPAT_H
#include <conio.h>
#define delay wc_delay_hidden /* Borland code uses "delay" as a variable */
#include <dos.h>
#undef delay
#include <math.h>
#define BLACK 0
#define BLUE 1
#define WHITE 15
#define _NOCURSOR 0
#define _NORMALCURSOR 2
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static void gotoxy(int x, int y) {           /* Borland is 1-based */
    union REGS r; r.h.ah = 2; r.h.bh = 0; r.h.dh = (unsigned char)(y - 1);
    r.h.dl = (unsigned char)(x - 1); int86(0x10, &r, &r);
}
static int wherey(void) {
    union REGS r; r.h.ah = 3; r.h.bh = 0; int86(0x10, &r, &r); return r.h.dh + 1;
}
static void clrscr(void) {
    union REGS r; r.x.ax = 0x0003; int86(0x10, &r, &r);
}
static void textcolor(int c) { (void)c; }
static void textbackground(int c) { (void)c; }
static void highvideo(void) {}
static void lowvideo(void) {}
static void _setcursortype(int t) { (void)t; }
#endif
#include "eqcompat.h"
#ifdef RB_MISSING
#include "rb_missing.h"
#endif
/* BPBC.C as printed has two unguarded debug pauses (print weights, wait for a
   key) per trial; for unattended runs a key is always "waiting". */
#define kbhit() 1
#define getch() (' ')
