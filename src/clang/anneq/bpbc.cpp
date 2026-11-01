// bpbc.cpp - mirrors src/borland/BP_INIT.H (globals, bp_init) and BPBC.C (main):
// the batch backpropagation equaliser simulator.
//   bpbc [TEST]    in a folder holding BP_CONST.TXT and WEIGHTS.TXT  -> ERRORS.TXT (per epoch),
//   BER.TXT (1995 file names e:\temp\bp_const.txt etc. with EQ_NOFIX=BP-PATHS, #17)
// BP_CONST.TXT: m_taps bits epochs snr runs nodes_0 nodes_1 nodes_2 learn_rate momentum
// A 3-layer tanh perceptron (nodes_1, nodes_2 hidden, one output) trained in batch mode with
// momentum on a fixed block of `bits` training vectors, `epochs` times per trial.
// Not ported: the conio banner and gotoxy(), the DEBUG / print_weights screen dumps (BP-DEBUG:
// they only wrote to the screen and waited for a key), bp_tidyup().
#include "eq_funcs.hpp"

using namespace eq;

// ============================== BP_INIT.H ===============================

static AnnStr ann;
static FirModel modelline;
static std::vector<TrainStr> train_vect;
static Signal errors;
static double snr;
static int m_taps;                                      // Number of taps in channel model
static int bits;                                        // Length of preamble in bits
static int runs;                                        // Number of trials to run
static std::array<int, 5> pn_shift = { 1, 1, 1, 1, 1 }; // PN sequence shift register
static int epochs;

// BP-PATHS (#17): plain file names in the current folder; 1995 used fixed e:\temp\ paths.
static std::string fname( const char *dos_name, const char *plain )
{
    return fixed()( Fix::BpPaths ) ? plain : dos_name;
}

static void bp_init()
{
    /* Load up simulation constants */

    ConstsFile consts_file =
        or_die( ConstsFile::open( fname( "e:\\temp\\bp_const.txt", "BP_CONST.TXT" ) ) );
    scan( consts_file, m_taps );
    scan( consts_file, bits );
    scan( consts_file, epochs );
    scan( consts_file, snr );
    scan( consts_file, runs );
    scan( consts_file, ann.nodes_0 );
    scan( consts_file, ann.nodes_1 );
    scan( consts_file, ann.nodes_2 );
    scan( consts_file, ann.learn_rate );
    scan( consts_file, ann.momentum );

    ann.nodes_3 = 1;

    /* Load in channel model weights */

    modelline = load_weights( fname( "e:\\temp\\weights.txt", "WEIGHTS.TXT" ), m_taps );

    /* Allocate memory for ann layers, each node a weight vector (+1: the bias input) */

    ann.layer_0.taps.assign( ann.nodes_0, NodeStr{} );
    ann.layer_0.has_bias = true;
    ann.layer_1 = make_layer( ann.nodes_1, ann.layer_0.size() );
    ann.layer_2 = make_layer( ann.nodes_2, ann.nodes_1 + 1 );
    ann.layer_3 = make_layer( ann.nodes_3, ann.nodes_2 + 1 );

    errors.assign( epochs, 0.0 );
    train_vect.assign( bits + static_cast<int>( std::ceil( ann.nodes_0 / 2.0 ) ), { 0.0, 0.0 } );
}

// The inputs a layer feeds to the layer above: its outputs and the bias node of -1 that 1995
// kept at layer_1[nodes_1] / layer_2[nodes_2].
static const Signal &with_bias( const Layer &layer, Signal &u )
{
    outputs( layer, u );
    u.push_back( -1.0 );
    return u;
}

// ================================ BPBC.C ================================

int main( int no_pars, char *par_txt[] )
{
    double error;
    double ber = 0; // Holds the bit error rate
    double desired; // Desired output from equaliser during testing
    int no_errors;  // Holds the number of errors occurred
    int test = 0;   // Holds whether to find BER for equaliser
    int index;
    double signal;
    double sum_sqr_err;
    Signal u0, u1, u2; // inputs of layers 1, 2, 3 (each with its bias)

    /* Initialise data */

    bp_init();

    /* Find out whether to test the BER of equaliser after training */

    test = params( { par_txt, static_cast<std::size_t>( no_pars ) } );

    /* Start simulations */

    program_seed();
    for ( int run_num = 0; run_num < runs; run_num++ )
    {
        std::print( stderr, "TRIAL:{:03}", run_num + 1 );
        trial_seed();

        /* Reset delay lines and network weights */

        for ( auto &tap : modelline )
        {
            tap.value = -1.0;
        }

        /* Set start bit */

        signal = start_bit( pn_shift );

        for ( TrainStr &t : train_vect ) // i < bits + ceil(nodes_0/2.0)
        {

            /* Add next preamble bit into channel model */

            t.desired = modelline[0].value = signal;

            /* Calculate signal and add to equalilser shift register */

            t.value = fir_output( modelline ) + noise( snr );

            /* Work out next value in PN sequence */

            signal = preamble_bit( pn_shift ); // 1995 inlined next_pn() here

            /* Shift the channel model FIR filter values along */

            shift_right_fir( modelline );
        }

        /* Initialise the network */

        // Range-for over nodes and weights: the 1995 i/j loops only reached each weight.
        for ( auto &node : ann.layer_1 )
        {
            for ( auto &w : node.weights )
            {
                w.weight = 0.3 * bc_rand() / RAND_MAX_BC; // 0.3*URAND as the macro expands
                w.delta = 0.0;
                w.delta_sf = 0.0;
            }
        }
        for ( auto &node : ann.layer_2 )
        {
            for ( auto &w : node.weights )
            {
                w.weight = 0.3 * bc_rand() / RAND_MAX_BC; // 0.3*URAND as the macro expands
                w.delta = 0.0;
                w.delta_sf = 0.0;
            }
        }
        for ( auto &node : ann.layer_3 )
        {
            for ( auto &w : node.weights )
            {
                w.weight = 0.3 * bc_rand() / RAND_MAX_BC; // 0.3*URAND as the macro expands
                w.delta = 0.0;
                w.delta_sf = 0.0;
            }
        }

        /* Set Bias nodes output */

        ann.layer_0.bias = -1.0; // layer_1 and layer_2's bias: with_bias()

        /* Start training */

        for ( int i = 0; i < epochs; i++ )
        {
            sum_sqr_err = 0.0;
            index = static_cast<int>( std::ceil( ann.nodes_0 / 2.0 ) );
            for ( int j = 0; j < bits; j++ )
            {
                for ( int k = 0; k < ann.nodes_0; k++ )
                {
                    ann.layer_0.taps[k].output = train_vect[index - k].value;
                }
                ffwd_sig( ann.layer_0.gather( u0 ), ann.layer_1 );
                ffwd_sig( with_bias( ann.layer_1, u1 ), ann.layer_2 );
                ffwd_sig( with_bias( ann.layer_2, u2 ), ann.layer_3 );
                error = train_vect[index].desired - ann.layer_3[0].output;
                sum_sqr_err += error * error;
                ann.layer_3[0].derivative = DER_SIGMOID( ann.layer_3[0].output ) * error;
                derivatives( ann.layer_2, ann.layer_3 );
                derivatives( ann.layer_1, ann.layer_2 );
                update_weights_sf( u2, ann.layer_3 );
                update_weights_sf( u1, ann.layer_2 );
                update_weights_sf( u0, ann.layer_1 );
                index++;
            }
            errors[i] += sum_sqr_err / bits;
            update_weights_fi( ann.layer_3, ann.learn_rate, ann.momentum );
            update_weights_fi( ann.layer_2, ann.learn_rate, ann.momentum );
            update_weights_fi( ann.layer_1, ann.learn_rate, ann.momentum );
        }

        /* Now work out BER by testing over 10000 bits */

        if ( test )
        {

            /* Reset error counter */

            no_errors = 0;

            /* Reset delay lines */

            for ( int i = 0; i < 10000; i++ )
            {

                /* Add next preamble bit into channel model */

                desired = modelline[0].value = random_bit();

                /* Calculate signal and add to equalilzer shift register */

                ann.layer_0.taps[0].output = fir_output( modelline ) + noise( snr );

                /* Feed the signals through the 3 layers */

                ffwd_sig( ann.layer_0.gather( u0 ), ann.layer_1 );
                ffwd_sig( with_bias( ann.layer_1, u1 ), ann.layer_2 );
                ffwd_sig( with_bias( ann.layer_2, u2 ), ann.layer_3 );

                /* Calculate output of equalizer and then work out its error */

                if ( sign( ann.layer_3[0].output ) != desired )
                {
                    no_errors++;
                }

                shift_right_layer( ann.layer_0.taps );
                shift_right_fir( modelline );
            }
            std::println( stderr, " LAST BER:{:05}", no_errors ); // "LAST BER:%05d"
            ber += static_cast<double>( no_errors ) / 10000;
        }
    }

    /* Store mean errors in a text file */

    save_doubles( errors, runs, fname( "e:\\temp\\errors.txt", "ERRORS.TXT" ), 15 ); // "%+2.15lf\n"

    /* Store mean bit error rate in a text file */

    save_doubles( std::array{ ber }, runs, fname( "e:\\temp\\ber.txt", "BER.TXT" ), 15 );

    return 0; // MINOR (#18): explicit exit status
}
