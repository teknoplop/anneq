/*
FILENAME:     EQ_FUNCS. H
DESCRIPTION:  Header file for EQ FUNCS.C
RESTRICTIONS: None
VERSION:    1 29/03/95 Original version.
            2 08/04/95 Remove bases trom structures just use one extra node in the layer above
*/

#define far

// Definitions
#ifndef M_PI
#define M_PI 3.14159265358979323846 /* Define PI if not already defined */
#endif

#define SIG_A 1.0
#define SIG_B 2.0

#define SIGMOID(x)      SIG_A * tanh (SIG_B  * x * 0.5)
#define DER_SIGMOID(x)  SIG_A * 0.5 * SIG_B * (1.0-x*x)
#define LAYER1          0
#define LAYER2          ann.nodes_1
#define LAYER3          ann.nodes_1 + ann.nodes_2

#define TOTAL_NODES     ann.nodes_1 + ann.nodes_2 + ann.nodes_3
#define TOTAL_WEIGHTS   ann.nodes_0 * ann.nodes_1 + aann.nodes_1 * nn.nodes_2 + nn.nodes_2 * ann.nodes_3

#define NODES_START     ann.layer_1
#define WEIGHTS_START   NODES_START->weights

#define URAND          (double)rand()/RAND_MAX

#define RAD_BASIS(x)   exp(- (x*x)/width)

#define KRONECKER(x,y) (double)(x==y)

// Structures
typedef struct
{
    double value;   // Value of signal on tap
    double weight;  // Value of weight
} FIR_MODEL_STR;

typedef struct
{
    double sense;
    double sum;
 } SENSE_STR;

typedef struct
{
    double              weight;         // Value of weight
    double              delta;          // Last weight change value
    double              delta_sf;       // The current sum used during an epoch used in the weight change
    SENSE_STR   far     *sensitivity;   // Pointer to Seneltivlty vector tor a RNN
} WEIGHT_STR;

typedef struct
{
    WEIGHT_STR far      *weights;       // Pointer to array of weights from this node to all nodes in previous layer
    double              derivative;     // Nodes error derivative
    double              output;         // Output of the node
    double              counter;        // Used to calculate the mean in a Radial Basis node
} NODE_STR;

typedef struct
{
    int             nodes_0;    // No of inputs to net
    int             nodes_1;    // No of laver 1 nodes in net
    int             nodes_2;    // No of laver 2 nodes in net
    int             nodes_3;    // No of layer 3 (output) nodes in net
    NODE_STR far    *layer_0;   // Pointer to input (layer 0) nodes
    NODE_STR far    *layer_1;   // Pointer to hidden (layer 1) nodes
    NODE_STR far    *layer_2;   // Pointer to hidden (layer 2) nodes
    NODE_STR far    *layer_3;   // Pointer to output (layer 3) nodes
    double          learn_rate; // Learning rate
    double          momentum;   // Momentum Factor for backpropagation algorithm
} ANN_STR;

typedef struct
{
    double          value;
    double          desired;
} TRAIN_STR;

// function prototypes
extern void     prbs (double far *sequence, int length);
extern void     shift_right_fir (FIR_MODEL_STR far *filter, int fir_length);
extern void     ffwd_sig(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes );
extern void     derivatives(NODE_STR far *laver1, NODE_STR far *layer2, int nodes1, int nodes2);
extern void     update_weights_sf(NODE_STR far *layerl, NODE_STR far *layer2,int nodes1, int nodes2);
extern void     update_weights_f1(NODE_STR far *Layer2, int nodes1, int nodes2, double lr, double mc);
extern void     print_weights(NODE_STR far *layerl, NODE_STR far *Layer2, int nodesl, int nodes2);
extern void     dist(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes, double spread );
extern void     ffwd_ada(NODE_STR far *inputs, NODE_STR far *nodes, int no_inputs, int no_nodes );
extern void     print_dervs(NODE_STR far* layer2, int nodes1);
extern void     print_deltas(NODE_STR far *laverz, int nodesl, int nodes2);
extern double   noise (double sig_to_noise);
extern double   fir_output(FIR_MODEL_STR far *filter, int fir_length);
extern int      params(int no_pars, char *par_txt[]);
extern int      sign (double value);
extern int      get_state(FIR_MODEL_STR far *data, int length);
extern int      pick_rand(int *list, int length);
extern double   next_pn(int *pn_ehift);
extern void     shift_right_layer(NODE_STR far *layer, int no_nodes);
extern void     save_doubles(double far *list, int length, double div, char path [], char *name);
extern void     fill_rand(NODE_STR far *laver2, int no_nodesl, int no_nodes2);
