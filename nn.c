/**
 * @file nn.c
 * @author Huang Yanzhen (yanzhenhuangwork@gmail.com)
 * @brief Neural Network constructions.
 * @version 0.1
 * @date 2024-12-25
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"
#include "xlinalg.h"
#include "nn.h"

Matrix *ReLU(Matrix *mat, long long i, long long j, va_list args)
{
    bool forward = va_arg(args, int);
    double x = mat_read(mat, i, j);

    if (x < 0)
        return mat_write(mat, i, j, 0);

    return mat_write(mat, i, j, forward ? x : 1);
}

Matrix *Sigmoid(Matrix *mat, long long i, long long j, va_list args)
{
    bool forward = va_arg(args, int);
    double x = mat_read(mat, i, j);

    double activation = 1 / (1 + exp(-x));
    if (forward)
    {
        return mat_write(mat, i, j, activation);
    }
    else
    {
        return mat_write(mat, i, j, activation * (1 - activation));
    }
}

double CrossEntropyLoss(Matrix *truth, Matrix *pred)
{
    if (truth->row != pred->row || truth->col != pred->col)
    {
        fprintf(stderr, "Calculate cross entropy failed."
                        "The shape of two vectors are not the same.");
    }

    // Matrix *loss = xmat_rand(truth->row, truth->col);
    double loss = 0;
    for (long long i = 0; i < truth->row; i++)
    {
        for (long long j = 0; j < truth->col; j++)
        {
            double truth_val = mat_read(truth, i, j);
            double pred_val = mat_read(pred, i, j);
            loss += -truth_val * log(pred_val) - (1 - truth_val) * log(1 - pred_val);
        }
    }

    return loss;
}

Matrix *softMax(Matrix *vec)
{
    if (vec->row != 1 && vec->col != 1)
    {
        fprintf(stderr, "Unable to softmax non-vector matrix.");
    }

    double sum = mat_elemSum(vec);

    for (long long i = 0; i < vec->row; i++)
    {
        for (long long j = 0; j < vec->col; j++)
        {
            mat_write(vec, i, j, exp(mat_read(vec, i, j)) / sum);
        }
    }

    return vec;
}

Matrix *_layer_init(Matrix *mat, long long row, long long col, va_list args)
{
    double scale = va_arg(args, double);
    double val = ((double)rand() / RAND_MAX) * 2.0 * scale - scale;
    return mat_write(mat, row, col, val);
}

Layer *nn_buildLayer(long long input, long long output)
{
    // Matrix* weights = xmat_rand(input, output);
    double scale = sqrt(6.0 / (input + output));
    Matrix *weights = xmat_traverse(xmat_rand(input, output), _layer_init, scale);
    if (weights == NULL)
    {
        printf("Build layer failed: Can't initialize weights.");
        return NULL;
    }

    Layer *layer = malloc(sizeof(Layer));
    if (layer == NULL)
    {
        printf("Build layer failed.");
        return NULL;
    }
    layer->weights = weights;
    return layer;
}

NN *nn_buildNN(
    long long input_size,
    long long hidden_size,
    long long ouptut_size,
    long long hidden_num,
    MatrixElementOperation activation)
{
    NN *nn = malloc(sizeof(NN));
    if (nn == NULL)
    {
        fprintf(stderr, "Build NN failed: Can't allocate memory for NN.");
        return NULL;
    }

    // Architecture of neural-network.
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->hidden_num = hidden_num;

    // Dynamic states during back propagation.
    nn->output_states = malloc((nn->hidden_num + 3) * sizeof(Matrix *));
    nn->delta_states = malloc((nn->hidden_num + 3) * sizeof(Matrix *));

    // Activation and loss function.
    nn->activation = activation;
    // nn->loss = loss;

    nn->layers = calloc(hidden_num + 2, sizeof(Layer *));
    if (nn->layers == NULL)
    {
        fprintf(stderr, "Build NN failed: Can't allocate memory for layers.");
        free(nn);
        return NULL;
    }

    // Input Layer
    Layer *input_layer = nn_buildLayer(input_size + 1, hidden_size + 1);
    if (input_layer == NULL)
    {
        fprintf(stderr, "Build NN failed: Can't allocate memory for input layer.");
        free(nn->layers);
        free(nn);
        return NULL;
    }

    nn->layers[0] = input_layer;

    // Hidden Layer
    for (long long i = 1; i < hidden_num + 1; i++)
    {
        Layer *hidden_layer = nn_buildLayer(hidden_size + 1, hidden_size + 1);
        if (hidden_layer == NULL)
        {
            fprintf(stderr, "Build NN failed: Can't allocate memory for hidden layer.");
            return NULL;
        }
        nn->layers[i] = hidden_layer;
    }

    // Output Layer
    Layer *output_layer = nn_buildLayer(hidden_size + 1, ouptut_size);
    if (output_layer == NULL)
    {
        fprintf(stderr, "Build NN failed: Can't allocate memory for output layer.");
        return NULL;
    }
    nn->layers[hidden_num + 1] = output_layer;
    return nn;
}

void nn_printNN(NN *nn)
{
    Layer **layers = nn->layers;
    for (long long k = 0; k < nn->hidden_num + 2; k++)
    {
        Matrix *weights = layers[k]->weights;
        if (weights == NULL)
        {
            fprintf(stderr, "Print NN failed: Can't print weights.");
            return;
        }
        printf("Layer %lld at %p:\n", k + 1, weights);
        mat_print(weights);
        printf("\n");
    }
}

Matrix *nn_forward(NN *nn, double *input, long long input_size)
{
    double *biased_input = malloc((input_size + 1) * sizeof(double));
    for (long long i = 0; i < input_size; i++)
    {
        biased_input[i] = input[i];
    }

    biased_input[input_size] = 1;

    Matrix *temp = mat_create(1, input_size + 1, biased_input);

    nn->output_states[0] = temp;

    for (long long layer = 0; layer < nn->hidden_num + 2; layer++)
    {
        Matrix *weights = nn->layers[layer]->weights;
        Matrix *product = mat_multmat(temp, weights);
        // Activation
        product = xmat_traverse(product, nn->activation, true);
        // Save output states
        nn->output_states[layer + 1] = product;
        temp = product;
    }

    return mat_transpose(temp);
}

NN *nn_backward(NN *nn, Matrix *target, Matrix *forward_output)
{
    // Target should be in column matrix.
    if (target->col != 1 || forward_output->col != 1)
    {
        fprintf(stderr, "Backward propagation failed: "
                        "Target and forward output should be a column matrix.");
        exit(1);
    }
    // Shape should match
    if (forward_output->row != target->row)
    {
        fprintf(stderr, "Backward propagation failed: "
                        "Forward output and target column matrices should have the same height.");
        exit(1);
    }

    // Learning rate
    double lr = 0.01;

    // delta-loss/delta-y
    Matrix *softmaxed_output = softMax(forward_output);
    Matrix *dLdy = mat_difmat(softmaxed_output, target);

    // Matrix *dLdz =

    // delta-y/delta-w
    Matrix *dydW = mat_transpose(nn->output_states[nn->hidden_num]); // (hidden_num + 2) - 1 - 1
    Matrix *dLdW = mat_multmat(dydW, mat_transpose(dLdy));
    nn->layers[nn->hidden_num + 1]->weights = mat_addmat(nn->layers[nn->hidden_num + 1]->weights, mat_multscal(dLdW, -lr));

    for (long long layer = nn->hidden_num + 1; layer >= 0; layer--)
    {
    }

    return nn;

    // // Total loss: Column Vector
    // Matrix *total_error = mat_addmat(target, mat_multscal(forward_output, -1));
    // // Matrix *total_error = nn->loss(target, forward_output);

    // // Running deltas: delta-loss/delta-y. Initialized to the deltas of the output layer.
    // Matrix *cur_deltas = xmat_traverse(total_error, nn->activation, false);
    // for (long long layer = nn->hidden_num + 1; layer >= 0; layer--)
    // {
    //     nn->delta_states[layer] = cur_deltas;
    //     Matrix *weights = nn->layers[layer]->weights;
    //     Matrix *previous_deltas = mat_multmat(weights, cur_deltas);
    //     cur_deltas = previous_deltas;
    // }

    // // Forward update the weights.
    // // Sequence doesn't matter.
    // for (long long layer = 0; layer < nn->hidden_num + 2; layer++)
    // {
    //     Matrix *cur_weights = nn->layers[layer]->weights; // Input weights of the current layer. (input size x output size)

    //     Matrix *previous_outputs = nn->output_states[layer]; // Input of the current layer.
    //     // mat_print(previous_outputs);

    //     Matrix *cur_outputs = nn->output_states[layer + 1]; // Output of the current layer

    //     Matrix *gradient = xmat_traverse(mat_transpose(cur_outputs), nn->activation, false); // Gradient of output: delta-loss/delta-y

    //     Matrix *next_deltas = nn->delta_states[layer]; // Deltas of the output

    //     Matrix *deltas_gradient = mat_pwpmat(next_deltas, gradient); // Deltas dot Gradient

    //     Matrix *_dw = mat_multmat(deltas_gradient, previous_outputs); // (Deltas dot Gradient) (out_sizex1) x prev_outputs (1xin_size)

    //     Matrix *dw = mat_transpose(mat_multscal(_dw, lr));

    //     nn->layers[layer]->weights = mat_addmat(nn->layers[layer]->weights, dw);
    // }

    // return nn;
}

// Copy this for debug.
// printf("~~~~~~ Layer %lld ~~~~~~\n", layer);

// printf("Cur weights:\n");
// mat_print(cur_weights);

// printf("Prev outputs:\n");
// mat_print(previous_outputs);

// printf("Cur outputs:\n");
// mat_print(cur_outputs);

// printf("Gradient:\n");
// mat_print(gradient);

// printf("Next Deltas:\n");
// mat_print(next_deltas);

// printf("dw:\n");
// mat_print(dw);
