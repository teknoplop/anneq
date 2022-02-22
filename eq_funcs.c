/*
FILENAME:     EQ_FUNCS.C
DESCRIPTION:  Provides useful functions for adaptive equaliser simulations.
RESTRICTIONS: None
VERSION:    1 29/03/95 Original version
            2 08/04/95 Remove biases from structures - just use one extra node in the layer above
*/

// include header files
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "eq_funcs.h"

int params(int no_pars, char *par_txt [])
{
    int     i;
    int     test = 0;
    char    *input_txt;
 
    if (no_pars > 1)
    {
        input_txt = par_txt[1];
        for (i=0; i< (int) strlen (input_txt); i++)
        {
            input_txt[i] = toupper(input_txt[i]);
        }

        if (!strcmp("TEST", input_txt))
        {
            test=1;
        }
    }
    return test;
}


/* 
FUNCTION PROTOTYPE: double noise (double sig_to noise)
DESCRIPTION:        Returns a Gaussian distributed number with a variance defined ov the inout
                    parameter sig_to noise (the SNR)
*/
double noise (double sig_to_noise)
{
    double rand1; // Holds two uniform random numbers

    /* Generate a uniform random number not equal to zero */
    while ((rand1= URAND) == 0.0);

    // Return a Gaussian random number
    return (sqrt (-2.0*pow (10.0, (double) (-sig_to_noise/10.0)) *log (rand1) ) *cos (2.0 * M_PI * URAND));
}

/*
FUNCTION PROTOTYPE: void prbs (double far *sequence, int length)
DESCRIPTION:        Fills the structure a sequence with 1's and -1's with equal probability of
                    occurance. The sequence length is defined by the input parameter length
*/
void prbs (double far *sequence, int length)
{
    int count;

    for (count=0; count<length; count++)
    {
        if (URAND > 0.5)
        {
            sequence [count] =1.0;
        }
        else
        {
             sequence [count] =-1.0;        
        }
    }
}

/*
FUNCTION PROTOTYPE: double fir_output (FIR_MODEL_STR far *filter, int fir_length)
DESCRIPTION:        Sums the product of each value held in the FIR structure at filter, with its
                    respective weight. The number of taps in the filter is given by the input
                    parameter fir_length
*/
double fir_output (FIR_MODEL_STR far *filter, int fir_length)
{
    int     i;          // Counter for a FOR loop
    double  prod = 0.0; /* Holds the current sum or products */

    for (i=0; i<fir_length;i++)
    {
        prod += filter[i].value * filter[i].weight;
    }
    return prod;
}

/*
FUNCTION PROTOTYPE: void shift_right_fir(FIR_MODEL_STR far *filter, int fir_length)
DESCRIPTION:        Shift the values on the delay line of the FIR filter structure at filter along to the
                    right one place. The length of the FIR filter is given by fir_length
*/
void shift_right_fir(FIR_MODEL_STR far *filter, int fir_length)
{
    int i;
    for (i= (fir_length-1); i>0; i--)
    {
        filter[i].value = filter[i-1].value;
    }
}

/*
FUNCTION PROTOTYPE: int sign (double value)
DESCRIPTION:        Returns the sign (+1 or -1) of the input parameter value.
*/
int sign (double value)
{
    int result = 1;

    if (value <0.0)
    {
        result = -1;
    }
    return result;
}

/*
FUNCTTON PROTOTYPE: void update_weights_sf( NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2 )
DESCRIPTION:        Update weights between layers 1 and 2 during epoch training with BP alogorithm
*/
void update_weights_sf( NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2 )
{
    int i, j;

    for (i=0; i<nodes2; i++)
    {
        for (j=0; j<nodes1; j++)
        {
            layer2[i].weights[j].delta_sf += layer2[i].derivative*layer1[j].output;
        }
    }
}

/*
FUNCTION PROTOTYPE: void update_weights_fi(NODE_STR far *layer2, int nodes1, int nodes2, double lr, double mc)
DESCRIPTION:        Update weights between layers 1 and 2 during epoch training with BP alogorithm
*/
void update_weights_fi(NODE_STR far *layer2, int nodes1, int nodes2, double lr, double mc)
{
    int i,j;

    for(i=0; i<nodes2; i++)
    {
        for( j=0; j<nodes1; j++)
        {
            // TODO: += followed by = looks sketchy
            layer2[i].weights[j].weight += layer2[i].weights[j].delta = layer2[i].weights[j].delta_sf * lr * (1.0 - mc) + mc * layer2[i].weights[j].delta;
            layer2[i].weights[j].delta_sf=0.0;
        }
    }
}

/*
FUNCTION PROTOTYPE: void derivatives(NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2)
DESCRIPTION:        Calculates local derivatives for nodes in laver 1
*/
void derivatives(NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2)
{
    int i,j;
    double sum;

    for (i=0; i<nodes1; i++)
    {   
        sum = 0.0;
        for (j=0; j<nodes2; j++)
        {
            sum += layer2[j].weights[i].weight * layer2[j].derivative;
        }

        layer1[i].derivative = sum * DER_SIGMOID(layer1[i].output) ;
    }
}

/*
FUNCTION PROTOTYPE: void ffwd_sig (NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes)
DESCRIPTION:        Feeds signal in layer 1 through to layer 2 with sigmoid activation functions
*/
void ffwd_sig (NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes)
{
    int     i,j;
    double  input;

    for (i=0; i<no_nodes; i++)
    {
        input = 0.0;
        for (j=0; j<no_inputs; j++)
        {
            input += inputs[j].output * nodes[i].weights[j].weight;
        }
        nodes[i].output = SIGMOID(input);
    }
}

/*
FUNCTION PROTOTYPE: void print_weights (NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2 )
DESCRIPTION:        Print weights and outputs between layers 1 and 2
*/
void print_weights (NODE_STR far *layer1, NODE_STR far *layer2, int nodes1, int nodes2 )
{
    int i,j;

    for (i=0; i < nodes1; i++)
    {
        for (j=0; j<nodes2; i++)
        {
            printf("%+01.5lf ", layer2[j].weights[i].weight);
        }
        printf(" | %+01.5lf\n", layer1[i].output);
    }

    for (i=0; i<nodes2; j++)
    {
        printf("=========");
    }
    printf("\n");

    for (i=0; i<nodes2; i++)
    {
        printf("%+01.5lf ", layer1[i].output) ;
    }
    printf("\n\n");
}

/*
FUNCTION PROTOTYPE: void print_dervs(NODE_STR far *layer2, int nodes1)
DESCRIPTION:        Prints deirivitives in layer 2
*/
void print_dervs(NODE_STR far *layer2, int nodes1)
{
    int i;
    
    for (i=0; i<nodes1; i++)
    {
        printf("%+01.5lf ", layer2[i].derivative);
    }
    printf("\n\n");
}

/*
FUNCTION PROTOTYPE: void print_deltas(NODE_STR far *layer2, int nodes1,int nodes2)
DESCRIPTION:        Prints changes in weights for layer
*/
void print_deltas(NODE_STR far *layer2, int nodes1,int nodes2)
{
    int i,j;

    for (i=0; i<nodes1; i++)
    {
        for(j=0; j<nodes2; j++)
        {
            printf("%+01.5lf ", layer2[j].weights[i].delta);
        }
        printf("\n");
    }
    printf("\n\n");
}

/*
FUNCTION PROTOTYPE: int get_state(FIR_MODEL_STR far *data, int length)
DESCRIPTION:        Work out which node in RBF layer should fire.
*/
int get_state(FIR_MODEL_STR far *data, int length)
{
    int state = 0;  // State that data is in
    int power;      // Holds current power welghting
    int i;          // Counter in FOR loop

    power = (int)pow( 2,length-1);

    for (i=0; i<length; i++)
    {
        if (data[i].value == 1)
        {
            state += power;
        }
        power /= 2;
    }
    return state;
}

/*
FUNCTION PROTOTYPE: int get_state(FIR_MODEL_STR far *data, int length)
DESCRIPTION:        Calculates Euclidean distance for layer nodes
*/
void dist(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes, double width)
{
    int i,j;
    double distance;

    for (i=0; i<no_nodes; i++)
    {
        distance = 0.0;
        
        for (j=0; j<no_inputs; j++)
        {
            distance += (nodes[i].weights[j].weight - inputs[j].output) * (nodes[i].weights[j].weight - inputs[j].output) ;
        }
        nodes[i].output = RAD_BASIS(sqrt (distance));
    }
}

/*
FUNCTION PROTOTYPE: void ffwd_ada(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes)
DESCRIPTION:        Feed signals forward through an ADADLINE layer
*/
void ffwd_ada(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes)
{
    int     i,j;
    double  input;

    for (i=0; i<no_nodes; i++ )
    {
        input = 0.0;
        for (j=0; j<no_inputs; j++)
        {
            input += inputs[j].output * nodes[i] .weights[j].weight;
        }
        nodes[i].output = input;
    }
}

// Ignore
int pick_rand( int *list, int length)
{
    int i;
    int index, value;

    index = (int) ( ( (double) rand () /RAND_MAX) * (length-1));

    value = list [index];
    for (i=index; i<length-1; i++)
    {
        list[i] = list[i+1];
    }

    return value;
}

// ignore
double next_pn(int *pn_shift)
{
    int i;
    double signal;

    signal = (double) (pn_shift[1] ^ pn_shift[4]) ;

    for (i=4; i>0; i--)
    {
        pn_shift[i] = pn_shift[i-1];
    }

    pn_shift[0] = (int) signal;

    if (signal == 0)
    {
        signal = -1.0;
    }
    return signal;
}


/*
FUNCTION PROTOTYPE: void shift_right_layer (NODE_STR far *layer, int no_nodes)
DESCRIPTION:        Shift values in layer one plave to right
*/
void shift_right_layer (NODE_STR far *layer, int no_nodes)
{
    int i;
    for ( i= no_nodes-1; i>0; i--)
    {
        layer[i].output = layer[i-1].output;
    }
}

/*
FUNCTION PROTOTYPE: void save_doubles (double far *list, int length, double div, char path[], char *name )
DESCRIPTION:        Saves an array of doubles to disk, first dividing by div.
*/
void save_doubles (double far *list, int length, double div, char path[], char *name )
{
    FILE    *output_file;
    char    filename[100];
    int     i;
    
    sprintf(filename,"%s%s",path,name);
    if ((output_file =fopen (filename, "wt "))== NULL )
    {
        printf( "File %s cannot be created", name);
        exit( 1 );
    }
    
    for (i=0; i<length; i++)
    {
        fprintf(output_file, "%+2.7lf\n",list[i]/div);
    }
    
    fclose(output_file);
}

/*
FUNCTION PROTOTYPE: void fill_rand(NODE_STR far *layer_2, int nodes_1, int nodes_2 )
DESCRIPTION:        Fills weights between layer 1 and layer2 with small random numbers
*/
void fill_rand(NODE_STR far *layer_2, int nodes_1, int nodes_2 )
{
    int i,j;

    for ( i=0; i<nodes_2; i++)
    {
        for(j=0; j<nodes_1; j++)
        {
            layer_2[i].weights[j].weight = URAND*0.1;
        }
    }
}