#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "eq_funcs.h"
#include "rb.h"

void main(int no_pars, char *par_txt[])
        {
        double                  ber = 0;                /* Holds the bit error rate */
    double                      signal;
        double                  output;
        time_t                  t;                              /* Variable used to seed random
number generator */
        char                    path[100];              /* Holds path that .exe file is in */
        int                     no_errors;              /* Holds the number of errors occurred */
        int                             test = 0;               /* Holds whether to find BER for
equaliser */
        int                             index;                  /* Holds which state the output
should be in */
        int                             i, j;                   /* Counters for FOR loops */
        int                             run_num;                /* Holds the current trial number
*/
        int                             delay_count;    /* Counts down whilst delaying
equalisation */

/* Get path name */

        strncpy(path, par_txt[0], strlen(par_txt[0]) - strlen("RB.EXE"));
        path[strlen(par_txt[0]) - strlen("RB.EXE")] = '\0';

/* Initialize LMS equalizer */

        rb_init(path);

/* Find out whether to test the BER of equaliser after training */

        test = params(no_pars, par_txt);

        for (run_num=0; run_num<runs; run_num++)
                {
                gotoxy(x, y);
                printf("TRIAL:%03d",run_num+1);

/* Seed the random number generator */

         time(&t);
                srand((unsigned)t);

/* Reset delay lines and equalizer weights */

                for(i=0; i<ann.nodes_1; i++)
                        {
                        for(j=0; j<ann.nodes_0; j++)
                                {
                                ann.layer_1[i].weights[j].weight = 0.0;
                                }
                        }

                for (i=0; i<m_taps; i++)
                        {
                        modelline[i].value = -1.0;
                        }

                for (i=0; i<ann.nodes_0; i++)
                        {
                        ann.layer_0[i].output  = -1.0;
                        }

                for (i=0; i<ann.nodes_1; i++)
                        {
                        ann.layer_1[i].counter  = 0;
                        }

/* Set start bit */

          signal = 1.0;

/* Remember how long to delay the desired response */

          delay_count = ann.nodes_0;
          gotoxy(x-10, y);
          printf("TRAINING ");

          for (bit_no = 0; bit_no<bits; bit_no++)
               {

/* Add next preamble bit into channel model */

               delayline[0].value = modelline[0].value = signal;

/* Calculate signal and add to equaliser shift register */

               ann.layer_0[0].output = fir_output(modelline, m_taps) + noise(snr);

/* Only start equalization if desired response has been delayed enough */

               if (delay_count == 0)
                    {

/* Get state that the output should be in */

                    index = get_state(delayline, memory);

/* Update the mean of the centre given to that state */

                    for(j=0; j<ann.nodes_0; j++)
                         {
                         ann.layer_1[index].weights[j].weight *=
ann.layer_1[index].counter;
                         ann.layer_1[index].weights[j].weight +=
ann.layer_0[j].output;
                         }

          ann.layer_1[index].counter++;

                    for(j=0; j<ann.nodes_0; j++)
                         {
                         ann.layer_1[index].weights[j].weight /=
ann.layer_1[index].counter;
               }
                    }
               else
                    {

/* Equalization has not begun so decrease delay counter */

                    delay_count--;
                    }

/* Shift channel model and equalizer shift registers one place to the right */

        signal = next_pn(pn_shift);
               shift_right_layer(ann.layer_0, ann.nodes_0);
               shift_right_fir(modelline, m_taps);
               shift_right_fir(delayline, memory);
               }

/* Now work out BER by testing over 10000 bits */

          if (test)
               {
               gotoxy(x-10, y);
               printf("TESTING    ");

/* Reset error counter */

               no_errors = 0;

/* Reset delay lines */

               for(i=0; i<10000; i++)
                    {

/* Add next preamble bit into channel model */

                    modelline[0].value = sign((2.0*URAND)-1.0);

/* Calculate signal and add to equalilzer shift register */

                    ann.layer_0[0].output = fir_output(modelline, m_taps) +
noise(snr);


/* Calculate output of equalizer and then work out its error */

          dist(ann.layer_0, ann.layer_1, ann.nodes_0, ann.nodes_1, ann.learn_rate);
                    ffwd_ada(ann.layer_1, ann.layer_2, ann.nodes_1, ann.nodes_2);

                    if (sign(ann.layer_2[0].output) != modelline[0].value)
                         {
                         no_errors++;
                         }

/* Shift channel model and equalizer shift registers one place to the right */

                    shift_right_layer(ann.layer_0, ann.nodes_0);
                    shift_right_fir(modelline, m_taps);
                    }
               gotoxy(x-25, y);
               printf("LAST BER:%05d", no_errors);
               ber += (double)no_errors/10000.0;
               }
          }

/* Store mean bit error rate in a text file */

     save_doubles(&ber, 1, runs, path, "BER.TXT");

/* Clean up memory */

     rb_tidyup();
     exit(0);
     }
