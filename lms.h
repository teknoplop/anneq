/*
FILENAME:     LMS.H
DESCRIPTION:  Header file for LMS.C
RESTRICTIONS: None
VERSION:    1 14/04/95 Original version
            2 08/04/95 Remove biases from structures - just use one extra node in the layer above
*/

//TODO: no prototypes for includes

/* Global variables */
extern ANN_STR             ann;             // Neural network data structure
extern FIR_MODEL_STR far   *modelline;      // Pointer to channel model shift regsiter
extern FIR_MODEL_STR far   *delayline;      // Pointer to channel model shift regsiter
extern double far          *errors;         // Pointer to output error vector
extern double far          *error_run;  
extern double              snr;             // Signal to noise ration in dB's
extern double              pass_threshold;
extern int                 taps;            // Number of equalizer taps
extern int                 m_taps;          // Number of taps in cannel model FIR filter
extern int                 bits;            // Length of preamble in bits
extern int                 runs;            // Number of trials to run
extern int                 x,y;             // Display co-ordinates
extern int                 pn_shift[5];     // PN sequence shift register
extern int                 delay;           // How lon to delay desired signal for

/* Function prototypes */
extern void                 lms_init( char par_txt[]);
extern void                 lms_tidyup(void);