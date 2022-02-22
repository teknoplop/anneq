// Mock borland conio lib

extern int noop();

#define _NOCURSOR
#define _NORMALCURSOR
#define WHITE
#define BLACK
#define _setcursortype(x) noop
#define textcolor(x) noop
#define textbackground(x) noop
#define gotoxy( x, y ) noop
#define highvideo noop 
#define lowvideo noop
#define cprintf printf
#define wherey noop