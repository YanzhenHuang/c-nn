#include "xlinalg.h"
/**
 * @brief Activation Function. 
 * 
 */
typedef double (*Activation)(double);

typedef struct {
    Matrix* weights;
} Layer;

typedef struct {
    long long input_size;
    long long hidden_size;
    long long output_size;
    long long hidden_num;
    Layer** layers;
    Activation activation;
} NN;


/**
 * @brief ReLU activation function.
 * 
 * @param x Input.
 * @return double 
 */
double ReLU(double x);

/**
 * @brief Sigmoid activation function.
 * 
 * @param x Input.
 * @return double 
 */
double Sigmoid(double x);

/**
 * @brief Build a neural network.
 * 
 * @param input_size Input size.
 * @param hidden_size Hidden size.
 * @param output_size Output size.
 * @param hidden_num Number of hidden layers.
 * @param activation Pointer to the activation function.
 * @return NN* 
 */
NN* nn_buildNN(long long input_size, long long hidden_size, long long output_size, long long hidden_num, Activation activation);

/**
 * @brief Print neural network.
 * 
 * @param nn Pointer to neural network struct.
 */
void nn_printNN(NN* nn);

/**
 * @brief Forward propagation.
 * 
 * @param nn Neural network struct pointer.
 * @param input Input array.
 * @param input_size Input size.
 * @return double* 
 */
Matrix* nn_forward(NN* nn, double* input, long long input_size);