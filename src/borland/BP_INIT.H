/*
        FILENAME:               BP_INIT.H

        DESCRIPTION:    Initialisation file for Backprop equaliser simulator.

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
#include "bpbc.h"

ANN_STR                                 ann;                    /* Neural network data structure
*/
FIR_MODEL_STR   far             *modelline;     /* Pointer to channel model shift register */
double                  far             *errors;                /* Pointer to output error vector
*/
double                                  snr;                    /* Signal to noise ratio in dB's
*/
TRAIN_STR               far             *train_vect;    /* Pointer to set of test vectors */
int                             far             *train_list;
int                                             taps;                   /* Number of equalizer
taps */
int                                             m_taps;                 /* Number of taps in
channel model FIR filter */
int                                             bits;                   /* Length of preamble in
bits */
int                                             runs;                   /* Number of trials to run
*/
int                                             x,y;                    /* Display co-ordinates */
int                                             pn_shift[5] = {1, 1, 1, 1, 1};          /* PN
sequence shift register */
int                                             epochs;

void bp_init(void)
        {
        FILE                    *consts_file;
        FILE                    *weights_file;
        int                             i;
    WEIGHT_STR far      *weight_ptr;
    NODE_STR   far      *node_ptr;

/* Display start banner */

        #ifndef DEBUG
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
        cprintf("%c BACKPROPAGATION EQUALIZER SIMULATOR v1.0
%c", (char)221, (char)222);
        cprintf("%c             FINAL YEAR PROJECT by [name omitted] IIIE 23/03/95
%c", (char)221, (char)222);
        textbackground(BLACK);
        for (i=0; i<80; i++)
                {
                cprintf("%c",(char)223);
                }
        textbackground(BLUE);
        y=wherey()-3;
        x=70;

        #else
                textmode(C4350);
        clrscr();
        printf("Initializing BP equalizer simulator - debugging mode\n");
        printf("Opening BP constants file\n");
        #endif


        if ((consts_file = fopen("e:\\temp\\bp_const.txt", "rt")) != NULL)
                {
                fscanf(consts_file, "%d%d%d%lf%d%d%d%d%lf%lf", &m_taps, &bits, &epochs, &snr,
&runs, &ann.nodes_0, &ann.nodes_1, &ann.nodes_2,
                                                                                &ann.learn_rate,
&ann.momentum);
                fclose(consts_file);
                }
        else
                {
                printf("File consts.txt not found\n");
                exit(1);
                }

        ann.nodes_3 = 1;

        #ifdef DEBUG
        printf("No of inputs:%02d\n",ann.nodes_0);
        printf("No of layer 1 nodes:%02d\n",ann.nodes_1);
        printf("No of layer 2 nodes:%02d\n",ann.nodes_2);
        printf("No of layer 3 nodes:%02d\n",ann.nodes_3);
        printf("Learn Rate:%+01.5lf\n",ann.learn_rate);
        printf("Momentum:%+01.5lf\n",ann.momentum);
        printf("No of preamble bits:%d\n", bits);
        printf("No of trials:%d\n", runs);
        printf("Signal to noise ratio:%lf dBs\n",snr);
        printf("Allocating memory for model FIR structure: %d * FIR_MODEL_STR\n", m_taps);
        #endif

        if ((modelline = (FIR_MODEL_STR far *)calloc(m_taps, sizeof(FIR_MODEL_STR))) == NULL)
                {
                printf("Calloc failiure on modelline");
                exit(1);
                }

        #ifdef DEBUG
        printf("Opening channel model weights file\nModel weights:");
        #endif

        if ((weights_file = fopen("e:\\temp\\weights.txt", "rt")) != NULL)
                {
                for(i=0; i<m_taps; i++)
                        {
                        fscanf(weights_file, "%lf\n", &modelline[i].weight);
                        #ifdef DEBUG
                        printf("%lf\t",modelline[i].weight);
                        #endif
                        }
                #ifdef DEBUG
                printf("\n");
                #endif
                fclose(weights_file);
                }
        else
                {
                printf("File weights.txt not found");
                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for input nodes: %d * bytes\n", (ann.nodes_0+1) *
sizeof(NODE_STR));
        #endif

        if ((ann.layer_0 = (NODE_STR  far *)calloc(ann.nodes_0+1, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on input nodes\n");

                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for layer 1 nodes: %d * bytes\n", (ann.nodes_1+1) *
sizeof(NODE_STR));
        #endif

        if ((ann.layer_1 = (NODE_STR  far *)calloc(ann.nodes_1+1, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on nodes\n");
                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for layer 2 nodes: %d * bytes\n", (ann.nodes_2+1) *
sizeof(NODE_STR));
        #endif

        if ((ann.layer_2 = (NODE_STR  far *)calloc(ann.nodes_2+1, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on nodes\n");
                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for layer 3 nodes: %d * bytes\n",
ann.nodes_3*sizeof(NODE_STR));
        #endif

        if ((ann.layer_3 = (NODE_STR  far *)calloc(ann.nodes_3, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on nodes\n");
                exit(1);
                }

        for(i=0; i<ann.nodes_1; i++)
                {
                #ifdef DEBUG
                printf("Allocating memory for layer 1 weights: %d * bytes\n", (ann.nodes_0+1) *
sizeof(WEIGHT_STR));
                #endif

                if ((ann.layer_1[i].weights = (WEIGHT_STR  far *)calloc(ann.nodes_0+1,
sizeof(WEIGHT_STR))) == NULL)
                        {
                        printf("Calloc failiure on nodes\n");
                        exit(1);
                        }
                }

        for(i=0; i<ann.nodes_2; i++)
                {
                #ifdef DEBUG
                printf("Allocating memory for layer 2 weights: %d * bytes\n", (ann.nodes_1+1) *
sizeof(WEIGHT_STR));
                #endif

                if ((ann.layer_2[i].weights = (WEIGHT_STR  far *)calloc((ann.nodes_1+1),
sizeof(WEIGHT_STR))) == NULL)
                        {
                        printf("Calloc failiure on nodes\n");
                        exit(1);
                        }
                }

        #ifdef DEBUG
        printf("Allocating memory for layer 3 weights: %d * bytes\n", (ann.nodes_2+1) *
sizeof(WEIGHT_STR));
        #endif

        if ((ann.layer_3[0].weights = (WEIGHT_STR  far *)calloc((ann.nodes_2+1),
sizeof(WEIGHT_STR))) == NULL)
                {
                printf("Calloc failiure on nodes\n");
                exit(1);

                }

        #ifdef DEBUG
        printf("Allocating memory for errors: %d * bytes\n", epochs * sizeof(double));
        #endif

        if ((errors = (double  far *)calloc(epochs, sizeof(double))) == NULL)
                {
                printf("Calloc failiure on errors\n");
                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for test vector: %d * bytes\n", (bits +
(int)ceil((double)ann.nodes_0/2)) * sizeof(TRAIN_STR));
        #endif

        if ((train_vect = (TRAIN_STR  far *)calloc(bits + (int)ceil((double)ann.nodes_0/2),
sizeof(TRAIN_STR))) == NULL)
                {
                printf("Calloc failiure on errors\n");
                exit(1);
                }

        #ifdef DEBUG
        printf("Allocating memory for test vector list: %d * bytes\n", bits * sizeof(int));
        #endif

        if ((train_list = (int far *)calloc(bits, sizeof(int))) == NULL)
                {
                printf("Calloc failiure on errors\n")/*?*/
                exit(1);
                }
        }

void bp_tidyup(void)
        {
        int i;

        free((void *)modelline);
        free((void *)errors);
        free((void *)train_vect);
    free((void *)train_list);
    free((void *)ann.layer_0);
    free((void *)ann.layer_1);
    free((void *)ann.layer_2);
        free((void *)ann.layer_3);

        for (i=0; i<ann.nodes_1; i++)
                {
                free((void *)ann.layer_1[i].weights);
                }

        for (i=0; i<ann.nodes_2; i++)
                {
                free((void *)ann.layer_2[i].weights);
                }
        free((void *)ann.layer_3[0].weights);

        #ifdef DEBUG
        printf("Left BP sim OK.\n\n");
        #endif
        }
