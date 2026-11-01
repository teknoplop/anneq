// rnn.cpp - mirrors src/borland/RNN_INIT.H (globals, rnn_init) and RNN.C (main):
// the recurrent (RTRL) equaliser simulator.
//   rnn [TEST]     in a folder holding RNN_CON.TXT and WEIGHTS.TXT
//   -> ERRORS.TXT, BER.TXT, FAIL.TXT (+ GAVEUP.TXT, ERRORS_ALL.TXT)
// RNN_CON.TXT: m_taps bits snr runs delay nodes_0 nodes_1 learn_rate
// nodes_1 fully recurrent tanh nodes; node 0 is the equaliser output. Each node sees
// layer_0 = [nodes_0 taps, bias -1, nodes_1 fed-back outputs].
// Not ported: the path from argv[0], the conio banner and gotoxy(), rnn_tidyup().
#include "eq_funcs.hpp"

using namespace eq;

// ============================== RNN_INIT.H ==============================

/* Global Variables */

static AnnStr ann;
static FirModel modelline;
static FirModel delayline;
static Signal errors;
static Signal error_run;
static Signal errors_all; // RETRY-CAP
// The SENSE_STR {sense, sum} vector each 1995 weight carried: sensitivity[j] of weight
// layer_1[k].weights[l] is sense[k, l, j]. Two 3-D Grids (mdspan views) in one block each.
static Grid<3> sense, sum;
static double snr;
static int m_taps;
static int bits;
static int runs;
static int delay;
static std::array<int, 5> pn_shift = { 1, 1, 1, 1, 1 };

static void rnn_init()
{
    /* Load up simulation constants */

    ConstsFile consts_file = or_die( ConstsFile::open( "RNN_CON.TXT" ) );
    scan( consts_file, m_taps );
    scan( consts_file, bits );
    scan( consts_file, snr );
    scan( consts_file, runs );
    scan( consts_file, delay );
    scan( consts_file, ann.nodes_0 );
    scan( consts_file, ann.nodes_1 );
    scan( consts_file, ann.learn_rate );

    /* Allocate memory for channel model FIR filter */
    /* Load in channel model weights */

    modelline = load_weights( "WEIGHTS.TXT", m_taps );

    /* Allocate memory for ann layers */

    ann.layer_0.taps.assign( ann.nodes_0, NodeStr{} );
    ann.layer_0.has_bias = true;
    ann.layer_0.feedback.assign( ann.nodes_1, NodeStr{} );

    /* Allocate each node a weight vector and each weight a sensitivity vector */

    const std::size_t n_all = ann.layer_0.size(); // 1995: nodes_0+nodes_1+1
    ann.layer_1 = make_layer( ann.nodes_1, n_all );
    sense = Grid<3>( ann.nodes_1, n_all, ann.nodes_1 );
    sum = Grid<3>( ann.nodes_1, n_all, ann.nodes_1 );

    /* Allocate memory for square error record */

    errors.assign( bits, 0.0 );
    errors_all.assign( bits, 0.0 );

    /* Allocate memory for square error */

    error_run.assign( bits, 0.0 );

    /* Allocate memory for desired signal delay line */

    delayline.assign( delay + 1, FirModelStr{ 0.0, 0.0 } );
}

// ================================ RNN.C =================================

int main( int no_pars, char *par_txt[] )
{
    double error;
    double ber = 0;
    double signal;
    double failiures = 0;
    double mean_err;
    int no_errors;
    int test;
    int delay_count;
    bool fail;
    Signal u; // layer_0 gathered
    const bool retry_cap = fixed()( Fix::RetryCap );
    int passed = 0, gaveup = 0, attempts;

    /* Initialise data */

    rnn_init();
    const int nodes_1 = ann.nodes_1;
    const std::size_t n_all = ann.layer_0.size();        // 1995: nodes_0+nodes_1+1
    const std::size_t fb = ann.layer_0.feedback_begin(); // 1995: nodes_0+1
    // RNN-FFWD (#10): feed the whole input vector; 1995 fed only taps and bias (nodes_0+1)
    const std::size_t n_in = fixed()( Fix::RnnFfwd ) ? n_all : fb;
    // RNN-KRON (#11): the delta_kj*u_l term sits inside the sum over the fed-back nodes; divide
    // it by nodes_1
    const bool kron_div = fixed()( Fix::RnnKron );

    /* Find out whether to test the BER of equaliser after training */

    test = params( { par_txt, static_cast<std::size_t>( no_pars ) } );

    /* Start simulations */

    program_seed();
    for ( int run_num = 0; run_num < runs; run_num++ )
    {
        std::print( stderr, "TRIAL:{:03}", run_num + 1 );
        fail = true;
        attempts = 0;
        while ( fail )
        {

            /* Seed the random number generator */

            trial_seed();
            attempts++;

            /* Reset channel model delay line*/

            for ( auto &tap : modelline )
            {
                tap.value = -1.0;
            }
            std::ranges::fill( sense.flat(), 0.0 );

            /* Initialise the network */

            fill_rand( ann.layer_1, n_all );

            /* Set bias node */

            ann.layer_0.bias = -1.0;

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

                    /* Copy delayed outputs to inputs */

                    for ( auto [input, node] : rv::zip( ann.layer_0.feedback, ann.layer_1 ) )
                    {
                        input.output = node.output;
                    }

                    /* Feed inputs through net */

                    ffwd_sig( ann.layer_0.gather( u ), ann.layer_1, n_in );

                    /* Calculate error and store square error */

                    error = delayline[delay].value - ann.layer_1[0].output;
                    error_run[bit_no - delay] = error * error;
                    if ( bit_no >= bits - 400 )
                    {
                        mean_err += error * error;
                    }

                    /* Calculate sensitivities */

                    // Plain index loops kept on purpose: k, l, j, i are the indices of the RTRL
                    // equation (weight w_kl, node j, fed-back node i) and appear in
                    // KRONECKER(k, j), u[l] and fb + i, so they read closer to the maths (and
                    // to RNN.C) than any range adaptor would. sum[k, l, j] is C++23 mdspan
                    // indexing for layer_1[k].weights[l].sensitivity[j].sum.
                    // C++23: zip pairs each fed-back input node with the node it copies.
                    for ( int k = 0; k < nodes_1; k++ )
                    {
                        for ( std::size_t l = 0; l < n_all; l++ )
                        {
                            for ( int j = 0; j < nodes_1; j++ )
                            {
                                sum[k, l, j] = 0.0;
                                for ( int i = 0; i < nodes_1; i++ )
                                {
                                    if ( kron_div )
                                    {
                                        sum[k, l, j] += ( ann.layer_1[j].weights[fb + i].weight *
                                                              sense[k, l, i] +
                                                          KRONECKER( k, j ) * u[l] / nodes_1 );
                                    }
                                    else
                                    {
                                        sum[k, l, j] += ( ann.layer_1[j].weights[fb + i].weight *
                                                              sense[k, l, i] +
                                                          KRONECKER( k, j ) * u[l] );
                                    }
                                }
                            }
                        }
                    }

                    for ( int k = 0; k < nodes_1; k++ )
                    {
                        for ( std::size_t l = 0; l < n_all; l++ )
                        {
                            for ( int j = 0; j < nodes_1; j++ )
                            {
                                sense[k, l, j] =
                                    DER_SIGMOID( ann.layer_1[j].output ) * ( sum[k, l, j] );
                            }
                        }
                    }

                    /* Update weights */

                    for ( int k = 0; k < nodes_1; k++ )
                    {
                        for ( std::size_t l = 0; l < n_all; l++ )
                        {
                            ann.layer_1[k].weights[l].weight +=
                                ann.learn_rate * error * sense[k, l, 0];
                        }
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
            }
            else
            {
                failiures++;
                if ( retry_cap && attempts >= max_attempts() )
                {
                    break;
                }
            }
        }
        for ( int i = 0; i < bits - delay; i++ )
        {
            if ( retry_cap )
            {
                errors_all[i] += error_run[i];
            }
            if ( !fail )
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

                /* Copy delayed outputs to inputs */

                for ( auto [input, node] : rv::zip( ann.layer_0.feedback, ann.layer_1 ) )
                {
                    input.output = node.output;
                }

                /* Feed inputs through net */

                ffwd_sig( ann.layer_0.gather( u ), ann.layer_1, n_in );

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
            std::println( stderr, " ERRORS:{:05}", no_errors );
            ber += static_cast<double>( no_errors ) / 10000;
        }
    }

    const std::size_t n_err = bits - delay;

    /* Store mean errors in a text file */

    if ( retry_cap )
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

    save_doubles( std::array{ ber }, runs, "BER.TXT" );

    /* Store mean number of failiures in a text file */

    save_doubles( std::array{ failiures }, runs, "FAIL.TXT" );

    return 0; // MINOR (#18): explicit exit status
}
