/*
        FILENAME:               RB_INIT.H

        DESCRIPTION:    Initialisation file for RBF equaliser simulator.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
                                        2       08/04/95        Remove biases from structures -
just use one extra node in the layer above.
*/
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include "eq_funcs.h"
#include "rb.h"


ANN_STR                         ann;                    /* Neural network data structure */
FIR_MODEL_STR   far     *modelline;     /* Pointer to channel model shift register */
FIR_MODEL_STR   far     *delayline;             /* Pointer to expected signal delay shift
register */
double                          snr;            /* Value for signal to noise ratio */
int                                     taps;                   /* Number of equalizer taps */
int                                     m_taps;                 /* Number of taps in channel model
FIR filter */
int                                     bits;                   /* Length of preamble in bits */
int                                     runs;           /* Number of trials to perform */
int                                     x,y;                    /* Display co-ordinates */
int                                     pn_shift[5]
                                                = {1, 1, 1, 1, 1};      /* PN sequence shift
register */
int                                     memory;                 /* Number of previous bits to
remember */

void rb_init(char path[])
        {
        FILE                    *consts_file;
        FILE                    *weights_file;
        char                    filename[100];
        int                             i;

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
        cprintf("%c RADIAL-BASIS EQUALISER SIMULATOR v1.0
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


/* Load up simulation constants */

        sprintf(filename, "%sRB_CON.TXT", path);
        if ((consts_file = fopen(filename, "rt")) != NULL)
                {
                fscanf(consts_file, "%d%d%lf%d%d%d%lf", &m_taps, &bits, &snr, &runs,
&ann.nodes_0, &ann.nodes_1, &ann.learn_rate);

                fclose(consts_file);
                }
        else
                {
                printf("\n\nFile %s not found", filename);
                exit(1);
                }

        ann.nodes_2 = 1;

        if ((modelline = (FIR_MODEL_STR far *)calloc(m_taps, sizeof(FIR_MODEL_STR))) == NULL)
                {
                printf("Calloc failiure on modelweights");
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
                printf("\n\nFile %s not found", filename);
                exit(1);
                }

/* Allocate memory for ann layers */

        if ((ann.layer_0 = (NODE_STR  far *)calloc(ann.nodes_0, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on input nodes\n");
                exit(1);
                }

        if ((ann.layer_1 = (NODE_STR  far *)calloc(ann.nodes_1, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on input nodes\n");
                exit(1);
                }

        if ((ann.layer_2 = (NODE_STR  far *)calloc(1, sizeof(NODE_STR))) == NULL)
                {
                printf("Calloc failiure on input nodes\n");
                exit(1);
                }

/* Allocate each node a weight vector */

        for(i=0; i<ann.nodes_1; i++)
                {
                if ((ann.layer_1[i].weights = (WEIGHT_STR far *)calloc(ann.nodes_0,
sizeof(WEIGHT_STR))) == NULL)
                        {
                        printf("Calloc failiure on weights\n");
                        exit(1);
                        }
                }
        if ((ann.layer_2[0].weights = (WEIGHT_STR far *)calloc(ann.nodes_1,
sizeof(WEIGHT_STR))) == NULL)
                {
                printf("Calloc failiure on weights\n");
                exit(1);
                }

/* Fill ADALINE weights vector */

        for(i=0; i<(ann.nodes_1/2); i++)
                {
                ann.layer_2[0].weights[i].weight = -1.0;

                }

        for(i=i;i<ann.nodes_1;i++)
                {
                ann.layer_2[0].weights[i].weight = 1.0;
                }

        memory = (int)(log10(ann.nodes_1)/log10(2));

        if ((delayline = (FIR_MODEL_STR far *)calloc(memory, sizeof(FIR_MODEL_STR))) == NULL)
                {
                printf("Calloc failiure on delayline");
                exit(1);
                }
        }

void rb_tidyup(void)
        {
        int i,j;

/* Free all allocated memory */

        for (i=0; i<ann.nodes_1; i++)
                {
                free((void *)ann.layer_1[i].weights);
                }
        free((void *)ann.layer_2[0].weights);
        free((void *)modelline);
        free((void *)delayline);
        free((void *)ann.layer_0);
    free((void *)ann.layer_1);
    free((void *)ann.layer_2);

/* Restore screen */

        textcolor(WHITE);
        textbackground(BLACK);
    lowvideo();
        _setcursortype(_NORMALCURSOR);
    printf("\n\n");
        }
