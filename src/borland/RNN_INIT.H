/*
        FILENAME:               RNN_INIT.H

        DESCRIPTION:    Initialisation file for RNN equaliser simulator.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
                                2       08/04/95        Remove biases from structures -
just use one extra node in the layer above.
*/
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "eq_funcs.h"
#include "rnnbc.h"

/* Global Variables */

ANN_STR                                 ann;                    /* Neural network data structure
*/
FIR_MODEL_STR   far             *modelline;     /* Pointer to channel model shift register */
FIR_MODEL_STR   far             *delayline;     /* Pointer to channel model shift register */
double                  far             *errors;                /* Pointer to output error vector
*/
double                  far             *error_run;             /* Pointer to output error vector
*/
double                                  snr;                    /* Signal to noise ratio in dB's
*/
int                                             taps;                   /* Number of equalizer
taps */
int                                             m_taps;                 /* Number of taps in
channel model FIR filter */
int                                             bits;                   /* Length of preamble in
bits */
int                                             runs;                   /* Number of trials to run
*/
int                                             x,y;                    /* Display co-ordinates */
int                                             delay;          /* How long to delay desired
signal for */
int                                             pn_shift[5] =
                                                        {1, 1, 1, 1, 1};        /* PN sequence
shift register */

void rnn_init(char path[])
        {
        FILE                    *consts_file;           /* Pointer to the input constants file */
        FILE                    *weights_file;          /* Pointer to the input channel model
weights file */
        char                    filename[100];          /* Temporary store for filename to open
*/
        int                             i,j;                    /* Counters for FOR loops
*/

/* Display start banner */

        printf("\n");
        _setcursortype(_NOCURSOR);
        textcolor(WHITE);
        textbackground(BLACK);
        highvideo();
        for (i=0; i<80; i++)
                {
                cprintf("%c",(char)220);
                }
        textbackground(BLUE);
        cprintf("%c RECURRENT NET EQUALIZER SIMULATOR v1.0
%c", (char)221, (char)222);
        cprintf("%c                     FINAL YEAR PROJECT by [name omitted] IIIE 08/04/95
%c", (char)221, (char)222);
        textbackground(BLACK);
        for (i=0; i<80; i++)
                {

                cprintf("%c",(char)223);
                }
        textbackground(BLUE);
        y=wherey()-3;
        x=70;

/* Load up simulation constants */

        sprintf(filename, "%sRNN_CON.TXT", path);
        if ((consts_file = fopen(filename, "rt")) != NULL)
                {
                fscanf(consts_file, "%d%d%lf%d%d%d%d%lf", &m_taps, &bits, &snr, &runs, &delay,
&ann.nodes_0, &ann.nodes_1, &ann.learn_rate);
                fclose(consts_file);
                }
        else
                {
                printf("\n\nFile %s not found\n", filename);
                exit(1);
                }

/* Allocate memory for channel model FIR filter */

        if ((modelline = (FIR_MODEL_STR far *)calloc(m_taps, sizeof(FIR_MODEL_STR))) == NULL)
                {
                printf("Calloc failiure on modelline");
                exit(1);
                }

/* Load in channel model weights */

        sprintf(filename, "%sWEIGHTS.TXT", path);
        if ((weights_file = fopen(filename, "rt")) != NULL)
                {
                for(i=0; i<m_taps; i++)
                        {
                        fscanf(weights_file, "%lf\n", &modelline[i].weight);
                        }
                fclose(weights_file);
                }
        else
                {
                printf("File %s not found", filename);
                exit(1);
                }

/* Allocate memory for ann layers */

        if ((ann.layer_0 = (NODE_STR  far *)calloc(ann.nodes_0 + ann.nodes_1 + 1,
sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on input nodes\n");
                exit(1);
                }

        if ((ann.layer_1 = (NODE_STR  far *)calloc(ann.nodes_1, sizeof(NODE_STR))) == NULL)
                {
                printf("\n\nCalloc failiure on nodes\n");
                exit(1);
                }

/* Allocate each node a weight vector and each weight a sensitivity vector */

        for(i=0; i<ann.nodes_1; i++)
                {
                if ((ann.layer_1[i].weights = (WEIGHT_STR far
*)calloc(ann.nodes_0+ann.nodes_1+1, sizeof(WEIGHT_STR))) == NULL)
                        {
                        printf("Calloc failiure on weights\n");
                        exit(1);
                        }
                for(j=0; j<ann.nodes_0+ann.nodes_1+1; j++)
                        {
                        if ((ann.layer_1[i].weights[j].sensitivity = (SENSE_STR far
*)calloc(ann.nodes_1, sizeof(SENSE_STR))) == NULL)
                                {

                                printf("Calloc failiure on nodes\n");
                                exit(1);
                                }
                        }
                }

/* Allocate memory for square error record */

        if ((errors = (double  far *)calloc(bits, sizeof(double))) == NULL)
                {
                printf("Calloc failiure on errors\n");
                exit(1);
                }

/* Allocate memory for square error */

        if ((error_run = (double  far *)calloc(bits, sizeof(double))) == NULL)
                {
                printf("Calloc failiure on errors\n");
                exit(1);
                }

/* Allocate memory for desired signal delay line */

        if ((delayline = (FIR_MODEL_STR far *)calloc(delay+1, sizeof(FIR_MODEL_STR))) == NULL)
                {
                printf("Calloc failiure on delayline");
                exit(1);
                }
        }

void rnn_tidyup(void)
        {
        int i,j;                /* Counters in FOR loops */

/* Free all allocated memory */


        for (i=0; i<ann.nodes_1; i++)
                {
                for(j=0; j<ann.nodes_0+ann.nodes_1; j++)
                        {
                        free((void *)ann.layer_1[i].weights[j].sensitivity);
                        }
                free((void *)ann.layer_1[i].weights);
                }

        free((void *)modelline);
        free((void *)delayline);
        free((void *)errors);
        free((void *)error_run);
        free((void *)ann.layer_0);
    free((void *)ann.layer_1);

/* Restore screen */

        textcolor(WHITE);
        textbackground(BLACK);
    lowvideo();
        _setcursortype(_NORMALCURSOR);
    printf("\n\n");
        }
