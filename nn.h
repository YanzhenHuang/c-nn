#include "xlinalg.h"

typedef struct {
    Matrix* weights;
} Layer;

typedef struct {
    Layer** layers;
    int activation;
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
 * @brief Build a perceptron layer.
 * 
 * @param input Input size.
 * @param output Output size.
 * @return Layer* 
 */
Layer* nn_buildLayer(long long input, long long output);

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
NN* nn_buildNN(long long input_size, long long hidden_size, long long output_size, long long hidden_num, int activation);