/*
        FILENAME:               RB.H

        DESCRIPTION:    Header file for RB.C.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
*/

extern  ANN_STR                         ann;                    /* Neural network data structure
*/
extern  FIR_MODEL_STR   far     *modelline;     /* Pointer to channel model shift register */
extern  FIR_MODEL_STR   far     *delayline;             /* Pointer to expected signal delay shift
register */
extern  double                          snr;            /* Value for signal to noise ratio */
extern  int                                     taps;                   /* Number of equalizer
taps */
extern  int                                     m_taps;                 /* Number of taps in
channel model FIR filter */
extern  int                                     bits;                   /* Length of preamble in
bits */
extern  int                                     runs;           /* Number of trials to perform */
extern  int                                     x,y;                    /* Display co-ordinates */
extern  int                                     pn_shift[5];    /* PN sequence shift register */
extern  int                                     memory;

extern  void    rb_init(char path[]);
extern  void    rb_tidyup(void);
