// TODO: incorrect filename extension in comments
// TODO: use xml or json for config (consts_file)
// TODO: single exit point

/*
FILENAME:     LMS_INIT.C
DESCRIPTION:  Initialisation file for LMS equaliser eimulator
RESTRICTIONS: None
VERSION:    1 29/03/95 Original version
            2 08/04/95 Remove biases from structures - just use one extra node in the layer above
*/

#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <string.h>
#include "eq_funcs.h"
#include "lms.h"

ANN_STR             ann;            // Neural network data structure
FIR_MODEL_STR far   *modelline;     // Pointer to channel model shift regsiter
FIR_MODEL_STR far   *delayline;     // Pointer to channel model shift regsiter
double far          *errors;        // Pointer to output error vector
double far          *error_run;  
double              snr;            // Signal to noise ration in dB's
double              pass_threshold;
int                 taps;           // Number of equalizer taps
int                 m_taps;         // Number of taps in cannel model FIR filter
int                 bits;           // Length of preamble in bits
int                 runs;           // Number of trials to run
int                 x,y;            // Display co-ordinates
int                 delay;          // How lon to delay desired signal for
int                 pn_shift[5] = { 1, 1, 1, 1, 1}; // PN sequence shift register

// TODO: separate presentation an I/O from business logic
void lms_init (char path[])
{
    FILE    *consts_file;
    FILE    *weights_file;
    char    filename[100];
    int     i;

    // Display start banner
    printf("\n");
    _setcursortype(_NOCURSOR);
    textcolor(WHITE);
    textbackground(BLACK);
    highvideo();
    
    for( i=0; i<80; i++)
    {
        cprintf("%c", (char) 220) ;
    }
    textbackground ( BLUE);
    // TODO: wrong banner?
    // TODO: carriage return at 80 chars in conio
    cprintf(   "\n%c ADALINE EQUALISER SIMULATOR V1.0                                             %c", (char) 221, (char) 222);
    cprintf(   "\n%c               FINAL YEAR PROJECT by TEKNO PLOP - EE 08/04/95                 %c\n", (char) 221,(char) 222);
    textbackground (BLACK);
    for (i=0; i<80; i++)
    {     
        cprintf("%c", (char) 223) ;
    }
    textbackground (BLUE) ;
    y=wherey () - 3 ;
    x=70;

    // Load up simulation constants
    sprintf( filename, "%sLMS_CON.TXT", path) ;
    if ((consts_file = fopen (filename, "rt" )) != NULL )
    {
        fscanf( consts_file, "%d%d%lf%d%d%d%lf", &m_taps, &bits, &snr, &runs, &delay, &ann.nodes_0, &ann.learn_rate);
        fclose( consts_file );
    }
    else
    {
        printf("\n\nFile %s not found", filename );
        exit(1);
    }

    ann.nodes_1  = 1;

    // Allocate memory for channel model FIR filter
    if ((modelline = (FIR_MODEL_STR far*) calloc (m_taps, sizeof (FIR_MODEL_STR))) == NULL)
    {
        printf ( "Calloc failiure on modelline");
        exit(1);
    }

    // Load in channel model weights
    sprintf(filename, "%sWEIGHTS.TXT", path);
    if ((weights_file = fopen( filename, "rt" )) != NULL)
    {
        for (i=0; i< m_taps; i++)
        {
            fscanf (weights_file, "%lf\n", &modelline[i].weight);
        }
        fclose (weights_file);
    }
    else
    {
        printf("\n\nFile %s not found", filename);
        exit(1);
    }

    // Allocate memory for ann layers
    if ((ann.layer_0 = (NODE_STR far *)calloc (ann.nodes_0+1, sizeof (NODE_STR))) == NULL)
    {
        printf ( "Calloc failure on input nodes\n");
        exit (1);
    }

    if ((ann.layer_1 = (NODE_STR far * )calloc(1, sizeof (NODE_STR) ) ) == NULL)
    {
        // TODO: wrong error message
        printf ( "Calloe failure on input nodes\n") ;
        exit (1);
    }

    // Allocate each node a weight vector
    if ((ann.layer_1[0].weights = (WEIGHT_STR far *)calloc(ann.nodes_0+1, sizeof (WEIGHT_STR) ) ) == NULL)
    {
        printf ("Calloc failure on weights\n") ;
        exit (1);
    }

    // Allocate memory for square error record
    if ((errors = (double far *)calloc (bits, sizeof (double) )) == NULL)
    {
        printf ("Calloc failure on errors\n") ;
        exit (1) ;
    }

    /* Allocate memory for square error */
    if ((error_run = (double far *)calloc (bits, sizeof (double) )) == NULL)
    {
        printf ( "Calloc failiure on errors\n");
        exit (1);
    }

    // Allocate memory for desired signal delay line
    if ((delayline = (FIR_MODEL_STR far*) calloc (delay+1, sizeof (FIR_MODEL_STR) ) ) == NULL)
    {
        printf ("Calloc failure on delayline");
        exit (1);
    }
}

void lms_tidyup (void)
{
    // Free all allocated memory
    free((void *)ann.layer_1[0].weights);
    free((void *)modelline);
    free((void *)delayline);
    free((void *)errors);
    free((void *)ann.layer_0);
    free((void *)ann.layer_1);

    // Restore screen
    textcolor (WHITE);
    textbackground(BLACK);
    lowvideo();
    _setcursortype (_NORMALCURSOR) ;
    printf("\n\n");
}