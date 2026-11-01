// rb.cpp - mirrors src/borland/RB_INIT.H (globals, rb_init) and RB.C (main):
// the radial basis function equaliser simulator.
//   rb [TEST]      in a folder holding RB_CON.TXT and WEIGHTS.TXT  -> BER.TXT
// RB_CON.TXT: m_taps bits snr runs nodes_0 nodes_1 width ("learn_rate" in 1995)
// One Gaussian node per channel state (nodes_1 = 2^memory). Training sets each centre to the
// mean of the received vectors seen in its state; the output node adds the nodes of the states
// whose newest bit is +1 and subtracts the others.
// Not ported: the path from argv[0], the conio banner and gotoxy(), rb_tidyup().
#include "eq_funcs.hpp"

using namespace eq;

// ============================== RB_INIT.H ===============================

static AnnStr ann;
static FirModel modelline;
static FirModel delayline;
static double snr;
static int m_taps; // Number of taps in channel model
static int bits;   // Length of preamble in bits
static int runs;   // Number of trials to perform
static std::array<int, 5> pn_shift = { 1, 1, 1, 1, 1 };
static int memory; // Number of previous bits to remember

static void rb_init()
{
    /* Load up simulation constants */

    ConstsFile consts_file = or_die( ConstsFile::open( "RB_CON.TXT" ) );
    scan( consts_file, m_taps );
    scan( consts_file, bits );
    scan( consts_file, snr );
    scan( consts_file, runs );
    scan( consts_file, ann.nodes_0 );
    scan( consts_file, ann.nodes_1 );
    scan( consts_file, ann.learn_rate ); // the Gaussian width

    ann.nodes_2 = 1;

    /* Load in channel model weights */

    modelline = load_weights( "WEIGHTS.TXT", m_taps );

    /* Allocate memory for ann layers */
    /* Allocate each node a weight vector */

    ann.layer_0.taps.assign( ann.nodes_0, NodeStr{} );
    ann.layer_1 = make_layer( ann.nodes_1, ann.nodes_0 ); // centres
    ann.layer_2 = make_layer( ann.nodes_2, ann.nodes_1 );

    /* Fill ADALINE weights vector */

    // -1 for the first half of the states, +1 after. C++23: zip(iota, range) numbers the
    // weights, as views::enumerate would (not in Apple's libc++).
    for ( auto [i, w] : rv::zip( rv::iota( 0 ), ann.layer_2[0].weights ) )
    {
        w.weight = i < ann.nodes_1 / 2 ? -1.0 : 1.0;
    }

    memory = static_cast<int>( std::log10( ann.nodes_1 ) / std::log10( 2 ) );
    delayline.assign( memory, FirModelStr{ 0.0, 0.0 } );
}

// ================================= RB.C =================================

int main( int no_pars, char *par_txt[] )
{
    double ber = 0;
    double signal;
    int no_errors;
    int test;
    int delay_count;
    int index;
    Signal u, u1; // layer_0 and layer_1 outputs as plain values, for dist() and ffwd_ada()

    /* Initialize LMS equalizer */

    rb_init();

    /* Find out whether to test the BER of equaliser after training */

    test = params( { par_txt, static_cast<std::size_t>( no_pars ) } );

    program_seed();
    for ( int run_num = 0; run_num < runs; run_num++ )
    {
        std::print( stderr, "TRIAL:{:03}", run_num + 1 );

        /* Seed the random number generator */

        trial_seed();

        /* Reset delay lines and equalizer weights */

        for ( auto &node : ann.layer_1 )
        {
            for ( auto &w : node.weights )
            {
                w.weight = 0.0;
            }
        }
        for ( auto &tap : modelline )
        {
            tap.value = -1.0;
        }
        for ( auto &node : ann.layer_0.taps )
        {
            node.output = -1.0;
        }
        for ( auto &node : ann.layer_1 )
        {
            node.counter = 0;
        }

        /* Set start bit */

        signal = start_bit( pn_shift );

        /* Remember how long to delay the desired response */

        delay_count = ann.nodes_0;

        for ( int bit_no = 0; bit_no < bits; bit_no++ )
        {

            /* Add next preamble bit into channel model */

            delayline[0].value = modelline[0].value = signal;

            /* Calculate signal and add to equaliser shift register */

            ann.layer_0.taps[0].output = fir_output( modelline ) + noise( snr );

            /* Only start equalization if desired response has been delayed enough */

            if ( delay_count == 0 )
            {

                /* Get state that the output should be in */

                index = get_state( delayline );

                /* Update the mean of the centre given to that state */

                for ( int j = 0; j < ann.nodes_0; j++ )
                {
                    ann.layer_1[index].weights[j].weight *= ann.layer_1[index].counter;
                    ann.layer_1[index].weights[j].weight += ann.layer_0.taps[j].output;
                }
                ann.layer_1[index].counter++;
                for ( int j = 0; j < ann.nodes_0; j++ )
                {
                    ann.layer_1[index].weights[j].weight /= ann.layer_1[index].counter;
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

        /* Now work out BER by testing over 10000 bits */

        if ( test )
        {

            /* Reset error counter */

            no_errors = 0;

            /* Reset delay lines */

            for ( int i = 0; i < 10000; i++ )
            {

                /* Add next preamble bit into channel model */

                modelline[0].value = random_bit();

                /* Calculate signal and add to equalilzer shift register */

                ann.layer_0.taps[0].output = fir_output( modelline ) + noise( snr );

                /* Calculate output of equalizer and then work out its error */

                dist( ann.layer_0.gather( u ), ann.layer_1, ann.learn_rate );
                ffwd_ada( outputs( ann.layer_1, u1 ), ann.layer_2 );
                if ( sign( ann.layer_2[0].output ) != modelline[0].value )
                {
                    no_errors++;
                }

                /* Shift channel model and equalizer shift registers one place to the right */

                shift_right_layer( ann.layer_0.taps );
                shift_right_fir( modelline );
            }
            std::println( stderr, " LAST BER:{:05}", no_errors );
            ber += static_cast<double>( no_errors ) / 10000.0;
        }
    }

    /* Store mean bit error rate in a text file */

    save_doubles( std::array{ ber }, runs, "BER.TXT" );

    return 0; // MINOR (#18): explicit exit status
}
