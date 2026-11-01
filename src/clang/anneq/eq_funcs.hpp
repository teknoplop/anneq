// eq_funcs.hpp - mirrors src/borland/EQ_FUNCS.H and EQ_FUNCS.C: the definitions, structures and
// shared functions of the four 1995 equaliser simulators, in the same order and with the same
// names (params, noise, fir_output, shift_right_fir, sign, update_weights_sf, update_weights_fi,
// derivatives, ffwd_sig, get_state, dist, ffwd_ada, next_pn, shift_right_layer, save_doubles,
// fill_rand). One .cpp per program, as in 1995: lms.cpp, rnn.cpp, rb.cpp, bpbc.cpp.
//
// The one deliberate change of layout: the 1995 code kept a layer's inputs as a flat NODE_STR
// array and appended extra "nodes" to it (the bias at layer_0[nodes_0], the fed-back RNN outputs
// at layer_0[nodes_0+1 ..]) and sized arrays by sums like nodes_0+nodes_1+1. Getting those
// offsets wrong caused #10, #12 and #13. Here the parts are named (Inputs: taps, bias, feedback)
// and joined in one place (Inputs::gather); every size comes from Inputs::size().
//
// Floating-point operations run in the same order as the C (macros expanded) and as
// src/anneq_fixed.py, so with the same EQ_SEED and -ffp-contract=off the output files are byte
// for byte those of the clang C builds and the Python port (src/tools/check_cpp.py). Every sum
// is a strict left-to-right fold (a for loop or std::ranges::fold_left),
// never std::reduce / transform_reduce, which may reassociate and change the rounding.
//
// C++23, building with Apple clang's libc++ as well as Homebrew LLVM's, so no
// std::views::enumerate (Apple's libc++ lacks it; zip with iota does the same job).
// Comments naming a standard (C++11/17/20/23) point at a modern feature replacing a 1995 C idiom.
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <expected>
#include <fstream>
#include <mdspan>
#include <memory>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace eq
{

namespace rv = std::views;

/* Definitions */

// constexpr variables and functions replace the #define macros: typed, each argument evaluated
// once. The expressions keep the macros' operator order exactly.
inline constexpr double SIG_A = 1.0;
inline constexpr double SIG_B = 2.0;
inline double SIGMOID( double x ) { return SIG_A * std::tanh( SIG_B * x * 0.5 ); }
constexpr double DER_SIGMOID( double x ) { return SIG_A * 0.5 * SIG_B * ( 1.0 - x * x ); }
constexpr double KRONECKER( int x, int y ) { return static_cast<double>( x == y ); }
inline double RAD_BASIS( double x, double width ) { return std::exp( -( x * x ) / width ); }

// ---------------------------------------------------------------------------------------------
// 2026 additions (not in EQ_FUNCS): fix tags and the Borland rand().
//
// Fixes: the tags of src/anneq_fixed.py. All on by default.
//   EQ_FIX=TAG,..    only these           EQ_NOFIX=TAG,..   all but these (EQ_NOFIX=ALL: the 1995
//   code)
// C++11 enum class: scoped names, no silent conversion to int. C++23 std::to_underlying turns a
// tag into its table index explicitly, where 1995 C would have used a bare #define'd int.
enum class Fix : std::uint8_t
{
    LmsFill,
    LmsBias,
    RnnFfwd,
    RnnKron,
    BpDebug, // have a src/fixes/*.patch
    RetryCap,
    SeedOnce,
    RandPreamble,
    BpPaths,
    Minor, // Python/C++ only
    Count_
};
inline constexpr std::array<std::string_view, std::to_underlying( Fix::Count_ )> kFixNames = {
    "LMS-FILL",  "LMS-BIAS",  "RNN-FFWD",      "RNN-KRON", "BP-DEBUG",
    "RETRY-CAP", "SEED-ONCE", "RAND-PREAMBLE", "BP-PATHS", "MINOR" };

inline std::string upper( std::string_view s )
{
    std::string u;
    for ( char c : s )
    {
        u += static_cast<char>( std::toupper( static_cast<unsigned char>( c ) ) );
    }
    return u;
}

class Fixes
{
  public:
    using Set = std::array<bool, kFixNames.size()>;

    Fixes()
    {
        Set want{}, drop{};
        bool want_any = false, want_all = false, drop_any = false, drop_all = false;
        // C++20: views::split replaces a strtok() loop over "A,B,C".
        auto parse = []( const char *env, Set &set, bool &any, bool &all )
        {
            const char *s = std::getenv( env );
            if ( !s )
            {
                return;
            }
            for ( auto part : std::string_view( s ) | rv::split( ',' ) )
            {
                std::string t = upper( std::string_view( part ) );
                std::erase( t, ' ' ); // C++20: erase-remove in one call
                if ( t.empty() )
                {
                    continue;
                }
                any = true;
                if ( t == "ALL" )
                {
                    all = true;
                    continue;
                }
                const auto it = std::ranges::find( kFixNames, t ); // C++20: whole-range find
                if ( it == kFixNames.end() )
                {
                    std::println( stderr, "unknown EQ_FIX tag: {}", t );
                    std::exit( 1 );
                }
                set[static_cast<std::size_t>( it - kFixNames.begin() )] = true;
            }
        };
        parse( "EQ_FIX", want, want_any, want_all );
        parse( "EQ_NOFIX", drop, drop_any, drop_all );
        if ( !want_any || want_all )
        {
            want.fill( true );
        }
        if ( drop_all )
        {
            want.fill( false );
        }
        // C++23: views::zip walks several ranges in step, a tuple of references per element.
        for ( auto [on, w, d] : rv::zip( on_, want, drop ) )
        {
            on = w && !d;
        }
    }
    bool operator()( Fix f ) const { return on_[std::to_underlying( f )]; }

  private:
    Set on_{};
};

inline const Fixes &fixed()
{
    static const Fixes f;
    return f;
}

inline int max_attempts() // RETRY-CAP: EQ_MAX_ATTEMPTS, default 10
{
    const char *s = std::getenv( "EQ_MAX_ATTEMPTS" );
    return s ? std::atoi( s ) : 10;
}

// Borland C++ 3.1 rand()/srand() with EQ_SEED, as src/openwatcom/eqcompat.c.
inline constexpr int RAND_MAX_BC = 0x7FFF;

class BorlandRand
{
  public:
    BorlandRand()
    {
        const char *e = std::getenv( "EQ_RAND" );
        use_libc_ = e && ( e[0] == 'l' || e[0] == 'L' );
        e = std::getenv( "EQ_SEED" );
        seed_base_ = e ? std::atol( e ) : -1;
    }
    int rand()
    {
        if ( use_libc_ )
        {
            return static_cast<int>( static_cast<double>( std::rand() ) / RAND_MAX * 0x7FFF );
        }
        seed_ = seed_ * 22695477u + 1u; // uint32_t wraps mod 2^32
        return static_cast<int>( ( seed_ >> 16 ) & 0x7FFF );
    }
    // EQ_SEED set: the k-th call seeds EQ_SEED + k instead of s.
    void srand( unsigned s )
    {
        if ( seed_base_ >= 0 )
        {
            s = static_cast<unsigned>( seed_base_ + seed_k_++ );
        }
        seed_ = s;
        std::srand( s );
    }

  private:
    std::uint32_t seed_ = 1;
    long seed_base_ = -1, seed_k_ = 0;
    bool use_libc_ = false;
};

inline BorlandRand &rng()
{
    static BorlandRand r;
    return r;
}

inline int bc_rand() { return rng().rand(); }
// URAND macro: (double)rand()/RAND_MAX
inline double URAND() { return static_cast<double>( bc_rand() ) / RAND_MAX_BC; }

// "Seed the random number generator": srand(time) before every trial/retry in 1995; not with
// SEED-ONCE, which seeds once at the start of main instead (EQ_SEED, k = 0, if set).
inline void trial_seed()
{
    if ( !fixed()( Fix::SeedOnce ) )
    {
        rng().srand( static_cast<unsigned>( std::time( nullptr ) ) );
    }
}
inline void program_seed()
{
    if ( fixed()( Fix::SeedOnce ) )
    {
        rng().srand( 1 );
    }
}

// ---------------------------------------------------------------------------------------------
/* Structures */

using Signal = std::vector<double>;

// FIR_MODEL_STR: value and weight per tap. A vector of these replaces the calloc'd array.
struct FirModelStr
{
    double value;  // Value of signal on tap
    double weight; // Value of weight
};
using FirModel = std::vector<FirModelStr>;

// WEIGHT_STR. The RNN sensitivity vector is not kept per weight here: rnn.cpp holds all the
// sensitivities in one 3-D std::mdspan-viewed block (Grid<3> below).
struct WeightStr
{
    double weight = 0.0;   // Value of weight
    double delta = 0.0;    // Last weight change value
    double delta_sf = 0.0; // The current sum during an epoch used in the weight change
};

// NODE_STR. std::vector<WeightStr> replaces the calloc'd "WEIGHT_STR far *weights".
struct NodeStr
{
    std::vector<WeightStr> weights; // weights from this node to all nodes in previous layer
    double derivative = 0.0;        // Nodes error derivative
    double output = 0.0;            // Output of the node
    double counter = 0.0;           // Used to calculate the mean in a Radial basis node
};

// A layer: what 1995 calloc'd as "NODE_STR far *layer_1" plus a weights calloc per node.
using Layer = std::vector<NodeStr>;
inline Layer make_layer( std::size_t no_nodes, std::size_t no_weights )
{
    return Layer( no_nodes, NodeStr{ std::vector<WeightStr>( no_weights ), 0.0, 0.0, 0.0 } );
}

// What layer_1 sees: 1995's layer_0[0 .. nodes_0+nodes_1], [taps..., bias, feedback...], with
// the bias at layer_0[nodes_0] and the feedback at layer_0[nodes_0+1 ..]. Those offsets caused
// #10, #12 and #13, so the three parts are named here and joined in one place, gather().
struct Inputs
{
    std::vector<NodeStr> taps; // tapped delay line: layer_0[0 .. nodes_0-1]
    bool has_bias = false;
    double bias = -1.0;            // layer_0[nodes_0].output
    std::vector<NodeStr> feedback; // fed-back outputs (RNN): layer_0[nodes_0+1 ..]

    std::size_t bias_count() const { return has_bias ? 1 : 0; }
    std::size_t size() const { return taps.size() + bias_count() + feedback.size(); }
    std::size_t feedback_begin() const { return taps.size() + bias_count(); } // 1995: nodes_0+1

    // u = the outputs of taps, then the bias, then the feedback: the order the 1995 sums ran in.
    const Signal &gather( Signal &u ) const
    {
        u.clear();
        for ( const auto &node : taps )
        {
            u.push_back( node.output );
        }
        if ( has_bias )
        {
            u.push_back( bias );
        }
        for ( const auto &node : feedback )
        {
            u.push_back( node.output );
        }
        return u;
    }
};

// Grid<R>: an R-dimensional array of doubles in one block, row-major, for the RNN
// sensitivities. C++23 std::mdspan is the multi-dimensional view over the block, and C++23
// multi-argument operator[] writes sense[k, l, j] for 1995's
// layer_1[k].weights[l].sensitivity[j].sense. Under libc++ hardening (make debug) mdspan checks
// every index against its extent.
template <std::size_t R> class Grid
{
  public:
    using Extents = std::dextents<std::size_t, R>;

    Grid() = default;
    template <class... N>
        requires( sizeof...( N ) == R )
    explicit Grid( N... n )
        : ext_( static_cast<std::size_t>( n )... ),
          data_( ( static_cast<std::size_t>( n ) * ... ), 0.0 )
    {
    }

    // The view is made on demand, so copying a Grid never leaves a dangling pointer.
    std::mdspan<double, Extents> view() { return { data_.data(), ext_ }; }
    double &operator[]( auto... i ) { return view()[i...]; }
    std::span<double> flat() { return data_; }

  private:
    Extents ext_{};
    std::vector<double> data_;
};

// The outputs of a layer's nodes as plain values (inputs to the layer above).
// C++20: ranges::transform with a projection-like lambda replaces the copy loop.
inline const Signal &outputs( const Layer &layer, Signal &u )
{
    u.resize( layer.size() );
    std::ranges::transform( layer, u.begin(), []( const auto &node ) { return node.output; } );
    return u;
}

// ANN_STR
struct AnnStr
{
    int nodes_0 = 0;         // No of inputs to net
    int nodes_1 = 0;         // No of layer 1 nodes in net
    int nodes_2 = 0;         // No of layer 2 nodes in net
    int nodes_3 = 0;         // No of layer 3 (output) nodes in net
    Inputs layer_0;          // input (layer 0) nodes
    Layer layer_1;           // hidden (layer 1) nodes
    Layer layer_2;           // hidden (layer 2) nodes
    Layer layer_3;           // output (layer 3) nodes
    double learn_rate = 0.0; // Learning rate
    double momentum = 0.0;   // Momentum Factor for backpropagation algorithm
};

// TRAIN_STR
struct TrainStr
{
    double value;
    double desired;
};

// ---------------------------------------------------------------------------------------------
// Functions, in EQ_FUNCS.C order. std::span (C++20) replaces the "far pointer + length" pairs:
// a view of a vector or array that knows its own size. Where 1995 looped "for (i = 0; i <
// no_nodes; i++)" only to reach layer[i], a range-based for (C++11) visits each node instead.

// params(): TEST as the first argument, any case. std::span gives argv a size.
inline int params( std::span<char *> par_txt )
{
    int test = 0;
    if ( par_txt.size() > 1 && upper( par_txt[1] ) == "TEST" )
    {
        test = 1;
    }
    return test;
}

// noise(): Gaussian noise of variance 10^(-snr/10). The cos argument keeps the unbracketed
// URAND macro's order, (2*pi*rand())/RAND_MAX.
inline double noise( double sig_to_noise )
{
    double rand1;

    /* Generate a uniform random number not equal to zero */

    while ( ( rand1 = URAND() ) == 0.0 )
    {
    }

    /* Return a Gaussian random number */

    return std::sqrt( -2.0 * std::pow( 10.0, -sig_to_noise / 10.0 ) * std::log( rand1 ) ) *
           std::cos( 2.0 * M_PI * bc_rand() / RAND_MAX_BC );
}

// fir_output(): prod += value * weight over the taps, in order.
inline double fir_output( const FirModel &filter )
{
    double prod = 0.0;
    for ( const auto &tap : filter )
    {
        prod += tap.value * tap.weight;
    }
    return prod;
}

// shift_right_fir(): each value one place to the right; [0] keeps its value for the caller to
// overwrite. The index is the point here (i takes from i-1), so the 1995 loop stays.
inline void shift_right_fir( FirModel &filter )
{
    for ( std::size_t i = filter.size() - 1; i > 0; i-- )
    {
        filter[i].value = filter[i - 1].value;
    }
}

constexpr int sign( double value ) { return value < 0.0 ? -1 : 1; }

// update_weights_sf(): add this vector's weight change to the epoch sum.
// C++23: views::zip walks each weight and its input in step, as a tuple of references.
inline void update_weights_sf( std::span<const double> layer1, Layer &layer2 )
{
    for ( auto &node : layer2 )
    {
        for ( auto [w, input] : rv::zip( node.weights, layer1 ) )
        {
            w.delta_sf += node.derivative * input;
        }
    }
}

// update_weights_fi(): apply the epoch's change with momentum
inline void update_weights_fi( Layer &layer2, double lr, double mc )
{
    for ( auto &node : layer2 )
    {
        for ( auto &w : node.weights )
        {
            w.delta = w.delta_sf * lr * ( 1.0 - mc ) + mc * w.delta;
            w.weight += w.delta;
            w.delta_sf = 0.0;
        }
    }
}

// derivatives(): error derivatives of layer1's nodes, back through layer2's weights. i picks
// weight i of every node above, so it stays an index.
inline void derivatives( Layer &layer1, const Layer &layer2 )
{
    for ( std::size_t i = 0; i < layer1.size(); i++ )
    {
        double sum = 0.0;
        for ( const auto &node : layer2 )
        {
            sum += node.weights[i].weight * node.derivative;
        }
        const double x = layer1[i].output;
        layer1[i].derivative = sum * SIG_A * 0.5 * SIG_B * ( 1.0 - x * x ); // sum*DER_SIGMOID(x)
    }
}

// ffwd_sig(): tanh nodes, each summing its first no_inputs weighted inputs, left to right.
inline void ffwd_sig( std::span<const double> inputs, Layer &nodes, std::size_t no_inputs )
{
    for ( auto &node : nodes )
    {
        double input = 0.0;
        for ( std::size_t j = 0; j < no_inputs; j++ )
        {
            input += inputs[j] * node.weights[j].weight;
        }
        node.output = SIGMOID( input );
    }
}
inline void ffwd_sig( std::span<const double> inputs, Layer &nodes )
{
    ffwd_sig( inputs, nodes, inputs.size() );
}

// get_state(): the last `length` transmitted bits as a binary number, newest most significant
inline int get_state( const FirModel &data )
{
    int state = 0;
    int power = static_cast<int>( std::pow( 2, static_cast<double>( data.size() ) - 1 ) );
    for ( const auto &tap : data )
    {
        if ( tap.value == 1 )
        {
            state += power;
        }
        power /= 2;
    }
    return state;
}

// dist(): Gaussian of the distance between the inputs and each node's centre.
// C++23: ranges::fold_left is a left fold (acc = op(acc, x) in order), so the squared distance
// sums in the 1995 order; zip pairs each centre coordinate with its input.
inline void dist( std::span<const double> inputs, Layer &nodes, double spread )
{
    for ( auto &node : nodes )
    {
        const double distance =
            std::ranges::fold_left( rv::zip( node.weights, inputs ), 0.0,
                                    []( double acc, auto wx )
                                    {
                                        auto [w, x] = wx;
                                        return acc + ( w.weight - x ) * ( w.weight - x );
                                    } );
        node.output = RAD_BASIS( std::sqrt( distance ), spread );
    }
}

// ffwd_ada(): linear (ADALINE) nodes
inline void ffwd_ada( std::span<const double> inputs, Layer &nodes, std::size_t no_inputs )
{
    for ( auto &node : nodes )
    {
        double input = 0.0;
        for ( std::size_t j = 0; j < no_inputs; j++ )
        {
            input += inputs[j] * node.weights[j].weight;
        }
        node.output = input;
    }
}
inline void ffwd_ada( std::span<const double> inputs, Layer &nodes )
{
    ffwd_ada( inputs, nodes, inputs.size() );
}

// next_pn(): 5-stage PN generator (31-bit period). std::array replaces int pn_shift[5].
inline double next_pn( std::array<int, 5> &pn_shift )
{
    const double signal = static_cast<double>( pn_shift[1] ^ pn_shift[4] );
    std::shift_right( pn_shift.begin(), pn_shift.end(), 1 );
    pn_shift[0] = static_cast<int>( signal );
    return signal == 0 ? -1.0 : signal;
}

// Next training symbol: next_pn() as in 1995, or a random bit (RAND-PREAMBLE, #16)
inline double random_bit() { return sign( ( 2.0 * URAND() ) - 1.0 ); } // sign((2.0*URAND)-1.0)
inline double preamble_bit( std::array<int, 5> &pn_shift )
{
    return fixed()( Fix::RandPreamble ) ? random_bit() : next_pn( pn_shift );
}
// "Set start bit": 1.0 in 1995
inline double start_bit( std::array<int, 5> &pn_shift )
{
    return fixed()( Fix::RandPreamble ) ? preamble_bit( pn_shift ) : 1.0;
}

// shift_right_layer(): each node's output one place to the right (i takes from i-1, so the
// 1995 index loop stays).
inline void shift_right_layer( std::vector<NodeStr> &layer )
{
    for ( std::size_t i = layer.size() - 1; i > 0; i-- )
    {
        layer[i].output = layer[i - 1].output;
    }
}

// die(): 1995's printf(...); exit(1). C++23 std::print formats with {} placeholders checked at
// compile time against the argument types, replacing printf's %-codes.
[[noreturn]] inline void die( std::string_view msg )
{
    std::print( stdout, "{}", msg );
    std::exit( 1 );
}

// save_doubles(): one "%+2.7lf" line per value (BPBC: "%+2.15lf"), each divided by div.
// "{:+.{}f}" is printf's "%+.*f": sign always, fixed point, precision from the next argument
// (printf's width 2 never mattered: the output is always wider).
// The FILE* is held by a unique_ptr, so fclose runs however the function is left.
// macOS printf writes NaN as "nan" despite '+'; std::format would write "+nan", so NaN is
// written by hand (ERRORS.TXT under RETRY-CAP when no trial passed).
inline void save_doubles( std::span<const double> list, double div, const std::string &name,
                          int precision = 7 )
{
    std::unique_ptr<std::FILE, decltype( &std::fclose )> output_file(
        std::fopen( name.c_str(), "w" ), &std::fclose );
    if ( !output_file )
    {
        die( "File " + name + " cannot be created" );
    }
    for ( double x : list )
    {
        const double v = x / div;
        if ( std::isnan( v ) )
        {
            std::println( output_file.get(), "nan" );
        }
        else
        {
            std::println( output_file.get(), "{:+.{}f}", v, precision );
        }
    }
}

// fill_rand(): small random weights, URAND*0.1, node by node. nodes_1 random numbers are drawn
// per node; draws beyond the weight vector (the 1995 LMS overrun, #12) are discarded.
inline void fill_rand( Layer &layer_2, std::size_t nodes_1 )
{
    for ( auto &node : layer_2 )
    {
        for ( std::size_t j = 0; j < nodes_1; j++ )
        {
            const double v = URAND() * 0.1;
            if ( j < node.weights.size() )
            {
                node.weights[j].weight = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reading the constants files (the fopen/fscanf blocks of *_INIT.H).
// C++23: std::expected<T, E> holds either a value or an error, so a read reports "not found"
// or "ended early" to its caller instead of printf-and-exit deep inside the read.
class ConstsFile
{
  public:
    static std::expected<ConstsFile, std::string> open( const std::string &filename )
    {
        ConstsFile f( filename );
        if ( !f.in_ )
        {
            return std::unexpected( "\n\nFile " + filename + " not found" );
        }
        return f;
    }
    // expected::transform (C++23) applies a function to the value and passes an error through.
    std::expected<int, std::string> read_int()
    {
        return word().transform( []( const std::string &w )
                                 { return static_cast<int>( std::stol( w ) ); } );
    }
    std::expected<double, std::string> read_double()
    {
        return word().transform( []( const std::string &w ) { return std::stod( w ); } );
    }

  private:
    explicit ConstsFile( const std::string &filename ) : in_( filename ) {}
    std::expected<std::string, std::string> word()
    {
        std::string w;
        if ( !( in_ >> w ) )
        {
            return std::unexpected( "\n\nFile ended early" );
        }
        return w;
    }
    std::ifstream in_;
};

// Unwrap a std::expected, or print its error and exit(1) as 1995 did: the simulators cannot go
// on without their constants.
template <class T> T or_die( std::expected<T, std::string> e )
{
    if ( !e )
    {
        die( e.error() );
    }
    return *std::move( e );
}

// fscanf(consts_file, "%d", &x) / "%lf": read the next field into x.
inline void scan( ConstsFile &f, int &x ) { x = or_die( f.read_int() ); }
inline void scan( ConstsFile &f, double &x ) { x = or_die( f.read_double() ); }

/* Load in channel model weights */
inline FirModel load_weights( const std::string &filename, int m_taps )
{
    auto weights_file = or_die( ConstsFile::open( filename ) );
    FirModel modelline( m_taps, FirModelStr{ 0.0, 0.0 } ); // calloc(m_taps, sizeof(FIR_MODEL_STR))
    for ( auto &tap : modelline )
    {
        scan( weights_file, tap.weight );
    }
    return modelline;
}

} // namespace eq
