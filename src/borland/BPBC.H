/*
        FILENAME:               BPBC.H

        DESCRIPTION:    Header file for EQ_FUNCS.C.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
*/

extern  ANN_STR                         ann;                    /* Neural network data structure
*/
extern  FIR_MODEL_STR   far     *modelline;     /* Pointer to channel model shift register */
extern  double                  far     *errors;                /* Pointer to output error vector
*/
extern  TRAIN_STR       far     *train_vect;    /* Pointer to set of test vectors */
extern  double                          snr;                    /* Signal to noise ratio in dB's
*/
extern  int                             far     *train_list;
extern  int                                     taps;                   /* Number of equalizer
taps */
extern  int                                     m_taps;                 /* Number of taps in
channel model FIR filter */
extern  int                                     bits;                   /* Length of preamble in
bits */
extern  int                                     runs;                   /* Number of trials to run
*/
extern  int                                     x,y;                    /* Display co-ordinates */
extern  int                                     pn_shift[5];
extern  int                                     epochs;

extern  void    bp_init(void);
extern  void    bp_tidyup(void);
