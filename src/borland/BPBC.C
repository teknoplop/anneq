/*
        FILENAME:               BPBC.C

        DESCRIPTION:    Main program for Backprop equaliser simulator.

        RESTRICTIONS:   None

        VERSION:                1       29/03/95        Original version.
                                        2       08/04/95        Remove biases from structures -
just use one extra node in the layer above.
*/
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "eq_funcs.h"
#include "bpbc.h"

void main(int no_pars, char *par_txt[])
        {
        FILE                    *output_file;   /* Pointer to the output file */
        FILE                    *error_file;
        double          error;
        double                  ber = 0;                /* Holds the bit error rate */
        double                  desired;                /* Desired output from equaliser during
testing */
        int                     no_errors;              /* Holds the number of errors occurred */
        int                             test = 0;               /* Holds whether to find BER for
equaliser */
        int                     i, j,k;
        int                             index;
        int                             run_num;
        double                  signal;
        char                    filename[50];
        time_t                  t;
        double                  sum_sqr_err;

/* Initialise data */

        bp_init();

/* Find out whether to test the BER of equaliser after training */

        test = params(no_pars, par_txt);

        #ifdef DEBUG
        if (test)
                {
                printf("BER TESTING REQUESTED\n");
                }
        #endif

/* Start simulations */

        for (run_num=0; run_num<runs; run_num++)
                {

                #ifndef DEBUG
                gotoxy(x, y);
                printf("TRIAL:%03d",run_num+1);
                #endif

                srand((unsigned) time(&t));

/* Reset delay lines and network weights */

                for (i=0; i<m_taps; i++)
                        {
                        modelline[i].value = -1.0;
                        }

/* Set start bit */

                signal = 1.0;

                #ifndef DEBUG
                gotoxy(x-10, y);
                printf("TRAINING ");
                #else
                printf("Generating Training Sequence\n");
                #endif

                for(i=0; i<(bits + ceil((double)ann.nodes_0/2.0)); i++)
                        {
/* Add next preamble bit into channel model */

                        train_vect[i].desired = modelline[0].value = signal;

/* Calculate signal and add to equalilser shift register */

                        train_vect[i].value = fir_output(modelline, m_taps) + noise(snr);

/* Work out next value in PN sequence */

            signal = pn_shift[1] ^ pn_shift[4];

                        for(j=4; j>0; j--)
                                {
                                pn_shift[j] = pn_shift[j-1];
                                }

                        pn_shift[0] = (int)signal;

                        if (signal == 0)
                                {
                                signal = -1.0;
                                }

/* Shift the channel model FIR filter values along */

                        shift_right_fir(modelline, m_taps);
                        }

        #ifdef DEBUG
                printf("Starting training ...\n");
                while(!kbhit());
                getch();
                clrscr();
                #endif
                                        gotoxy(1, 1);
                        print_weights(ann.layer_1, ann.layer_0, ann.nodes_0+1, ann.nodes_1);
                        print_weights(ann.layer_2, ann.layer_1, ann.nodes_1+1, ann.nodes_2);
                        print_weights(ann.layer_3, ann.layer_2, ann.nodes_2+1, ann.nodes_3);
                        printf("\n");
                        while(!kbhit());
                        getch();

/* Initialise the network */

                for(i=0; i<ann.nodes_1; i++)
                        {
                        for(j=0; j<ann.nodes_0+1; j++)
                                {
                                ann.layer_1[i].weights[j].weight = 0.3* URAND
/*(2.4/ann.nodes_0)*/;
                                ann.layer_1[i].weights[j].delta = 0.0;
                                ann.layer_1[i].weights[j].delta_sf = 0.0;
                                }
                        }

                for(i=0; i<ann.nodes_2; i++)
                        {
                        for(j=0; j<ann.nodes_1+1; j++)
                                {
                                ann.layer_2[i].weights[j].weight = 0.3*URAND
/*(2.4/ann.nodes_1)*/;
                                ann.layer_2[i].weights[j].delta = 0.0;
                                ann.layer_2[i].weights[j].delta_sf = 0.0;
                                }

                        }

                for(i=0; i<ann.nodes_3; i++)
                        {
                        for(j=0; j<ann.nodes_2+1; j++)
                                {
                                ann.layer_3[i].weights[j].weight = 0.3 * URAND
/*(2.4/ann.nodes_1)*/;
                                ann.layer_3[i].weights[j].delta = 0.0;
                                ann.layer_3[i].weights[j].delta_sf = 0.0;
                                }
                        }

/* Set Bias nodes output */

                ann.layer_0[ann.nodes_0].output = -1.0;
                ann.layer_1[ann.nodes_1].output = -1.0;
                ann.layer_2[ann.nodes_2].output = -1.0;

                        gotoxy(1, 1);
                        print_weights(ann.layer_1, ann.layer_0, ann.nodes_0+1, ann.nodes_1);
                        print_weights(ann.layer_2, ann.layer_1, ann.nodes_1+1, ann.nodes_2);
                        print_weights(ann.layer_3, ann.layer_2, ann.nodes_2+1, ann.nodes_3);
                        printf("\n");
                        while(!kbhit());
                        getch();

/* Start training */

                for(i=0; i<epochs; i++)
                        {
                        sum_sqr_err = 0.0;
                        index =ceil((double)ann.nodes_0/2.0);
                        for(j=0; j<bits; j++)
                                {
                 for(k=0; k<ann.nodes_0; k++)
                                        {
                                        ann.layer_0[k].output = train_vect[index-k].value;
                                        }
                                ffwd_sig(ann.layer_0, ann.layer_1, ann.nodes_0+1, ann.nodes_1);
                                ffwd_sig(ann.layer_1, ann.layer_2, ann.nodes_1+1, ann.nodes_2);
                                ffwd_sig(ann.layer_2, ann.layer_3, ann.nodes_2+1, ann.nodes_3);
                                error = train_vect[index].desired - ann.layer_3[0].output;
                                sum_sqr_err += error*error;
                                ann.layer_3[0].derivative = DER_SIGMOID(ann.layer_3[0].output) *
error;
                                derivatives(ann.layer_2, ann.layer_3, ann.nodes_2, ann.nodes_3);
                                derivatives(ann.layer_1, ann.layer_2, ann.nodes_1, ann.nodes_2);
                                update_weights_sf(ann.layer_2, ann.layer_3, ann.nodes_2+1,
ann.nodes_3);
                                update_weights_sf(ann.layer_1, ann.layer_2, ann.nodes_1+1,
ann.nodes_2);
                                update_weights_sf(ann.layer_0, ann.layer_1, ann.nodes_0+1,
ann.nodes_1);
                                index++;
                                }
                        #ifdef DEBUG
            clrscr();
                        gotoxy(1, 1);
                        print_weights(ann.layer_1, ann.layer_0, ann.nodes_0+1, ann.nodes_1);
                        print_weights(ann.layer_2, ann.layer_1, ann.nodes_1+1, ann.nodes_2);
                        print_weights(ann.layer_3, ann.layer_2, ann.nodes_2+1, ann.nodes_3);
                        printf("\n");
                        while(!kbhit());
                        getch();
                        #endif
                        errors[i] += sum_sqr_err/bits;
                        update_weights_fi(ann.layer_3, ann.nodes_2+1, ann.nodes_3,
ann.learn_rate, ann.momentum);
                        update_weights_fi(ann.layer_2, ann.nodes_1+1, ann.nodes_2,
ann.learn_rate, ann.momentum);
                        update_weights_fi(ann.layer_1, ann.nodes_0+1, ann.nodes_1,
ann.learn_rate, ann.momentum);
                        }

/* Now work out BER by testing over 10000 bits */

                if (test)
                        {
                        #ifndef DEBUG
                        gotoxy(x-10, y);
                        printf("TESTING    ");
                        #else
                        printf("Starting Testing ...\n");
                        #endif

/* Reset error counter */

                        no_errors = 0;

/* Reset delay lines */

                        for(i=0; i<10000; i++)
                                {

/* Add next preamble bit into channel model */

                                desired = modelline[0].value = sign(2.0*URAND-1.0);

/* Calculate signal and add to equalilzer shift register */

                                ann.layer_0[0].output = fir_output(modelline, m_taps) +
noise(snr);

/* Feed the signals through the 3 layers */

                                ffwd_sig(ann.layer_0, ann.layer_1, ann.nodes_0+1, ann.nodes_1);
                                ffwd_sig(ann.layer_1, ann.layer_2, ann.nodes_1+1, ann.nodes_2);
                                ffwd_sig(ann.layer_2, ann.layer_3, ann.nodes_2+1, ann.nodes_3);

/* Calculate output of equalizer and then work out its error */

                                if (sign(ann.layer_3[0].output) != desired)
                                        {
                                        no_errors++;
                                        }

                                for(j=ann.nodes_0-1; j>0; j--)
                                        {
                                        ann.layer_0[j].output = ann.layer_0[j-1].output;
                                        }
                                shift_right_fir(modelline, m_taps);
                                }
                        #ifndef DEBUG
                        gotoxy(x-25, y);
                        printf("LAST BER:%05d", no_errors);
                        #else
                        printf("LAST BER:%05d\n", no_errors);
                        #endif
                        ber += (double)no_errors/10000;
                        }
                }
        #ifdef DEBUG
        clrscr();
        print_weights(ann.layer_1, ann.layer_0, ann.nodes_0, ann.nodes_1);
        print_weights(ann.layer_2, ann.layer_1, ann.nodes_1, ann.nodes_2);
        print_weights(ann.layer_3, ann.layer_2, ann.nodes_2, ann.nodes_3);
        printf("\n");
        while(!kbhit());
        getch();
        #endif

/* Store mean errors in a text file */

        if ((output_file = fopen("e:\\temp\\errors.txt", "wt")) == NULL)
                {
                printf("File errors.txt cannot be created");
                exit(1);
                }

        for (i=0; i<epochs; i++)
                {

                fprintf(output_file,"%+2.15lf\n", errors[i]/runs);
                }

        fclose(output_file);

/* Store mean bit error rate in a text file */

        if ((output_file = fopen("e:\\temp\\ber.txt", "wt")) == NULL)
                {
                printf("File ber.txt cannot be created");
                exit(1);
                }


        fprintf(output_file,"%+2.15lf\n", ber/runs);

        fclose(output_file);
        bp_tidyup();

        }
