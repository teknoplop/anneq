/* FTLENAME:
SUMS
DESCRIPTION:
Main
program for LMS equaliser
gimulator
RESTRICTIONS:
None
VERSION:
29/03/95
original version.
02 /04/05
Remov
1286
from
structures
Just use one extra node in the laver
above
*/
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "eq_funcs.h"
#include "lms.h"

void main (int no_pars, char *par_txt[])
{
    double  error;       // Current error fo equaliser's output
    double  ber = 0;     // Holds the bit error
    double  signal;      // Next bit to transmit
    double  failiures = 0;
    double  mean_err;
    time_t  t;           // variable use to seed random number generator
    char    path[100];  // Holds path that .exe file is in
    int     no_errors;  // Holds the number of errors occurred
    int     test;       // Holds whether to find BER for equaliser
    int     i, j, bit_no; // counters in FOR loops
    int     run_num;    // current trial number
    int     delay_count; // counts down whilst delaying equalization
    int     fail;

    // Get path name
    strncpy( path, par_txt[0], strlen(par_txt[0]) - strlen ("LMS.EXE"));
    path[strlen(par_txt[0]) - strlen ("LMS.EXE")] = '\0';

    // Initialize LMS equalizer
    lms_init( path );

    // Find out whether to test the BER of equalizer after training
    test = params (no_pars, par_txt) ;

    // Start simulations
    for (run_num=0; run_num<runs; run_num++)
    {
        gotoxy(x, y);
        printf("TRIAL: %03d", run_num + 1);
    
        fail = 1;

        while (fail)
        {
            // seed the random number generator
            time( &t );
            srand((unsigned)t);

            // Reset delay lines and equalizer weights
            for (i = 0; i < m_taps; i++)
            {
               modelline[i].value = -1.0; 
            }

            for (i = 0; i < ann.nodes_0; i++)
            {
                ann.layer_0[i].output = -1.0;
            }

            // initialise the network
            fill_rand( ann.layer_1, ann.nodes_0+ann.nodes_1+1, ann.nodes_1);

            // set start bit
            signal = 1.0;

            // remember how long to delay the desired response
            delay_count = delay;

            gotoxy( x - 10, y );
            printf( "TRAINING ");
            mean_err = 0.0;

            for ( bit_no = 0; bit_no < bits; bit_no++)
            {
                // Add next preamble bit into channel model
                delayline[0].value = modelline[0].value = signal;

                // Calculate signal and add to equaliser shift register
                ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr);

                // Only start equalisation if desired response has been delayed enough
                if (delay_count == 0)
                {  
                    // Feed inputs through net
                    ffwd_ada( ann.layer_0, ann.layer_1, ann.nodes_0+1, ann.nodes_1);

                    // Calculate error and store square error
                    error = delayline[delay].value - ann.layer_1[0].output;
                    error_run[bit_no-delay] = error*error;
                    if (bit_no>= bits-400)
                    {
                        mean_err+=error*error;
                    }

                    // Adjust equalizer weights according to LMS algorithm
                    for ( j = 0; j < ann.nodes_0 + 1; j++)
                    {
                        ann.layer_1[0].weights[j].weight += 2.0 * ann.learn_rate * error * ann.layer_0[j].output;
                    }
                }
                else
                {
                    // Equalisation has not begun so decrease delay counter
                    delay_count--;
                }

                // Shift channel model and equalizer shift registers one place to right
                signal = next_pn( pn_shift );
                shift_right_layer( ann.layer_0, ann.nodes_0);
                shift_right_fir(modelline, m_taps);
                shift_right_fir(delayline, delay+1);

                // TODO: what's this do?
                if ((mean_err/400) < 0.6 )
                {
                    fail = 0;
                    gotoxy( x - 4, y + 1 );
                    printf("%1.3lf PASSED", mean_err/400);
                }
                else
                {
                    gotoxy(x-4, y+1);
                    printf("%1.3lf FAILED", mean_err/400);
                    // putch((char)7);
                    failiures++;
                }
            }

            for ( i = 0; i < bits - delay; i++)
            {
                errors[i] += error_run[i];
            }
            
            // Now work out BER by testing over 10000 bits
            if ( test )
            {
                gotoxy( x - 10, y);
                printf( "TESTING   ");

                // Reset error counter 
                no_errors = 0;
                for (i = 0; i < 10000; i++)
                {
                    // Add next preamble bit into channel model
                    delayline[0].value = modelline[0].value = sign((2.0*URAND) -1.0);

                    // Calculate signal and add to equaliser shift register
                    ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr);

                    // Feed inputs through net
                    ffwd_ada( ann.layer_0, ann.layer_1, ann.nodes_0 + 1, ann.nodes_1);

                    // calculate output of equaliser and then work out its error
                    if (sign(ann.layer_1[0].output) != delayline[delay].value)
                    {
                        no_errors++;
                    }

                    // Shift channel model and eqalizer shift register one place to the right
                    shift_right_layer(ann.layer_0, ann.nodes_0);
                    shift_right_fir(modelline, m_taps);
                    shift_right_fir(delayline, delay+1);
                }
            gotoxy( x-25, y);
            printf("ERRORS:%05d", no_errors);
            ber += (double)no_errors/10000;
            }
        }
        // Store mean errors in a text file
        save_doubles( errors, bits-delay, runs, path, "ERRORS.TXT");

        // Store mean bit error rate in a text file
        save_doubles( &ber, 1, runs, path, "BER.TXT");

        // Store mean number of failures in a text file
        save_doubles( &failiures, 1, runs, path, "FAIL.TXT");

        // Clean up memory
        lms_tidyup();
        exit(0); // TOOD: just return 0?
    }

}

