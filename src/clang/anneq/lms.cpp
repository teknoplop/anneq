// lms.cpp - mirrors src/borland/LMS_INIT.H (globals, lms_init) and LMS.C (main):
// the LMS (ADALINE) equaliser simulator.
//   lms [TEST]     in a folder holding LMS_CON.TXT and WEIGHTS.TXT
//   -> ERRORS.TXT (mean square error per bit), BER.TXT, FAIL.TXT (+ GAVEUP.TXT, ERRORS_ALL.TXT)
// LMS_CON.TXT: m_taps bits snr runs delay nodes_0 learn_rate
// Not ported: the path taken from argv[0] (files are read from the current folder), the conio
// banner and gotoxy() screen positions, and lms_tidyup() (vectors free themselves).
#include "eq_funcs.hpp"

using namespace eq;

// ============================== LMS_INIT.H ==============================

// Global variables, as in 1995. std::vector replaces each calloc'd far pointer.
static AnnStr ann;         // Neural network data structure
static FirModel modelline; // channel model shift register
static FirModel delayline; // desired signal delay line
static Signal errors;      // output error vector
static Signal error_run;
static Signal errors_all; // RETRY-CAP: every trial's last attempt
static double snr;        // Signal to noise ratio in dB's
static int m_taps;        // Number of taps in channel model FIR filter
static int bits;          // Length of preamble in bits
static int runs;          // Number of trials to run
static int delay;         // How long to delay desired signal for
static std::array<int, 5> pn_shift = { 1, 1, 1, 1, 1 }; // PN sequence shift register

static void lms_init()
{
    /* Load up simulation constants */

    ConstsFile consts_file = or_die( ConstsFile::open( "LMS_CON.TXT" ) );
    scan( consts_file, m_taps );
    scan( consts_file, bits );
    scan( consts_file, snr );
    scan( consts_file, runs );
    scan( consts_file, delay );
    scan( consts_file, ann.nodes_0 );
    scan( consts_file, ann.learn_rate );

    ann.nodes_1 = 1;

    /* Allocate memory for channel model FIR filter */
    /* Load in channel model weights */

    modelline = load_weights( "WEIGHTS.TXT", m_taps );

    /* Allocate memory for ann layers */

    ann.layer_0.taps.assign( ann.nodes_0, NodeStr{} );
    ann.layer_0.has_bias = true;
    ann.layer_0.bias = 0.0; // 1995: never set (#13); LMS-BIAS sets it to -1 each trial

    /* Allocate each node a weight vector */

    ann.layer_1 = make_layer( ann.nodes_1, ann.layer_0.size() ); // 1995: nodes_0+1 weights

    /* Allocate memory for square error record */

    errors.assign( bits, 0.0 );
    errors_all.assign( bits, 0.0 );

    /* Allocate memory for square error */

    error_run.assign( bits, 0.0 );

    /* Allocate memory for desired signal delay line */

    delayline.assign( delay + 1, FirModelStr{ 0.0, 0.0 } );
}

// ================================ LMS.C =================================

int main( int no_pars, char *par_txt[] )
{
    double error;   // Current error of equaliser's output
    double ber = 0; // Holds the bit error rate
    double signal;  // Next bit to transmit
    double failiures = 0;
    double mean_err;
    int no_errors;   // Holds the number of errors occurred
    int test;        // Holds whether to find BER for equaliser
    int delay_count; // Counts down whilst delaying equalisation
    bool fail;
    Signal u; // layer_0 gathered: [taps, bias]
    const bool retry_cap = fixed()( Fix::RetryCap );
    int passed = 0, gaveup = 0, attempts;

    /* Initialize LMS equalizer */

    lms_init();

    /* Find out whether to test the BER of equaliser after training */

    test = params( { par_txt, static_cast<std::size_t>( no_pars ) } );

    /* Start simulations */

    program_seed();
    for ( int run_num = 0; run_num < runs; run_num++ )
    {
        std::print( stderr, "TRIAL:{:03}", run_num + 1 ); // printf("TRIAL:%03d", run_num+1)

        fail = true;
        attempts = 0;

        while ( fail )
        {

            /* Seed the random number generator */

            trial_seed();
            attempts++;

            /* Reset delay lines and equalizer weights */

            // C++11 range-for: "for (i=0; i<m_taps; i++) modelline[i].value = ..." where i only
            // reaches the element
            for ( auto &tap : modelline )
            {
                tap.value = -1.0;
            }
            for ( auto &node : ann.layer_0.taps )
            {
                node.output = -1.0;
            }
            if ( fixed()( Fix::LmsBias ) )
            {
                ann.layer_0.bias = -1.0;
            }

            /* Initialise the network */

            if ( fixed()( Fix::LmsFill ) )
            {
                fill_rand( ann.layer_1, ann.layer_0.size() );
            }
            else // 1995: nodes_0+nodes_1+1 random numbers for nodes_0+1 weights (#12)
            {
                fill_rand( ann.layer_1, ann.nodes_0 + ann.nodes_1 + 1 );
            }

            /* Set start bit */

            signal = start_bit( pn_shift );

            /* Remember how long to delay the desired response */

            delay_count = delay;

            mean_err = 0.0;

            for ( int bit_no = 0; bit_no < bits; bit_no++ )
            {

                /* Add next preamble bit into channel model */

                delayline[0].value = modelline[0].value = signal;

                /* Calculate signal and add to equaliser shift register */

                ann.layer_0.taps[0].output = fir_output( modelline ) + noise( snr );

                /* Only start equalisation if desired response has been delayed enough */

                if ( delay_count == 0 )
                {

                    /* Feed inputs through net */

                    ffwd_ada( ann.layer_0.gather( u ), ann.layer_1 );

                    /* Calculate error and store square error */

                    error = delayline[delay].value - ann.layer_1[0].output;
                    error_run[bit_no - delay] = error * error;
                    if ( bit_no >= bits - 400 )
                    {
                        mean_err += error * error;
                    }

                    /* Adjust equalizer weights according to LMS algorithm */

                    for ( std::size_t j = 0; j < u.size(); j++ )
                    {
                        ann.layer_1[0].weights[j].weight += 2.0 * ann.learn_rate * error * u[j];
                    }
                }
                else
                {

                    /* Equalization has not begun so decrease delay counter */

                    delay_count--;
                }

                /* Shift channel model and equalizer shift registers one place to the right */

                signal = preamble_bit( pn_shift );
                shift_right_layer( ann.layer_0.taps );
                shift_right_fir( modelline );
                shift_right_fir( delayline );
            }
            if ( ( mean_err / 400 ) < 0.6 )
            {
                fail = false;
                std::print( stderr, " {:.3f} PASSED", mean_err / 400 ); // "%1.3lf PASSED"
            }
            else
            {
                std::print( stderr, " {:.3f} FAILED", mean_err / 400 );
                failiures++;
                if ( retry_cap && attempts >= max_attempts() ) // RETRY-CAP: give up on the trial
                {
                    break;
                }
            }
        }
        for ( int i = 0; i < bits - delay; i++ )
        {
            if ( retry_cap ) // every trial's last attempt, passed or not
            {
                errors_all[i] += error_run[i];
            }
            if ( !fail ) // without RETRY-CAP every trial passes eventually
            {
                errors[i] += error_run[i];
            }
        }
        if ( !fail )
        {
            passed++;
        }
        else
        {
            gaveup++;
        }

        /* Now work out BER by testing over 10000 bits */

        if ( test )
        {

            /* Reset error counter */

            no_errors = 0;

            for ( int i = 0; i < 10000; i++ )
            {

                /* Add next preamble bit into channel model */

                delayline[0].value = modelline[0].value = random_bit();

                /* Calculate signal and add to equaliser shift register */

                ann.layer_0.taps[0].output = fir_output( modelline ) + noise( snr );

                /* Feed inputs through net */

                ffwd_ada( ann.layer_0.gather( u ), ann.layer_1 );

                /* Calculate output of equalizer and then work out its error */

                if ( sign( ann.layer_1[0].output ) != delayline[delay].value )
                {
                    no_errors++;
                }

                /* Shift channel model and equalizer shift registers one place to the right */

                shift_right_layer( ann.layer_0.taps );
                shift_right_fir( modelline );
                shift_right_fir( delayline );
            }
            std::println( stderr, " ERRORS:{:05}", no_errors ); // "ERRORS:%05d"
            ber += static_cast<double>( no_errors ) / 10000;
        }
    }

    // span::first(n) passes the first n values, as 1995 passed (errors, bits-delay).
    const std::size_t n_err = bits - delay;

    /* Store mean errors in a text file */

    if ( retry_cap ) // curve = mean over the trials that passed; BER over all trials
    {
        const Signal curve = passed ? errors : Signal( bits, NAN );
        save_doubles( std::span( curve ).first( n_err ), passed ? passed : 1, "ERRORS.TXT" );
        save_doubles( std::array{ static_cast<double>( gaveup ) }, runs, "GAVEUP.TXT" );
        save_doubles( std::span( errors_all ).first( n_err ), runs, "ERRORS_ALL.TXT" );
    }
    else
    {
        save_doubles( std::span( errors ).first( n_err ), runs, "ERRORS.TXT" );
    }

    /* Store mean bit error rate in a text file */

    save_doubles( std::array{ ber }, runs, "BER.TXT" ); // 1995: save_doubles(&ber, 1, ...)

    /* Store mean number of failiures in a text file */

    save_doubles( std::array{ failiures }, runs, "FAIL.TXT" );

    return 0; // MINOR (#18): explicit exit status
}
