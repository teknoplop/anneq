/*
        FILENAME:               RNNBC.H

        DESCRIPTION:    Header file for RNNBC.C.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
                                2       08/04/95        Remove biases from structures -
just use one extra node in the layer above.
*/

extern  ANN_STR                         ann;                    /* Neural network data structure
*/
extern  FIR_MODEL_STR   far     *modelline;     /* Pointer to channel model shift register */
extern  FIR_MODEL_STR   far     *delayline;     /* Pointer to channel model shift register */
extern  double                  far     *errors;                /* Pointer to output error vector
*/
extern  double                  far     *error_run;             /* Pointer to output error vector
*/
extern  double                          snr;                    /* Signal to noise ratio in dB's
*/
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
extern  int                                     delay;

extern  void    rnn_init(char par_txt[]);
extern  void    rnn_tidyup(void);
