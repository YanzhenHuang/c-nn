#include "xlinalg.h"
typedef struct {
    Matrix* weights;
} Layer;

typedef struct {
    long long input_size;
    long long hidden_size;
    long long output_size;
    long long hidden_num;
    Layer** layers;
    MatrixElementOperation activation;
} NN;


/**
 * @brief ReLU activation function.
 * 
 * @param x Input.
 * @return double 
 */
Matrix* ReLU(Matrix* mat, long long i, long long j, va_list args);

/**
 * @brief Sigmoid activation function.
 * 
 * @param x Input.
 * @return double 
 */
Matrix* Sigmoid(Matrix* mat, long long i, long long j, va_list args);

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
NN* nn_buildNN(long long input_size, long long hidden_size, long long output_size, long long hidden_num, MatrixElementOperation activation);

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

/**
 * @brief Backward propagation.
 * 
 * @param nn Neural network struct pointer.
 * @param forward_output Output of the forward propagation.
 * @param target Desired output.
 * @return Matrix* 
 */
Matrix* nn_backward(NN* nn, Matrix* forward_output, Matrix* target);