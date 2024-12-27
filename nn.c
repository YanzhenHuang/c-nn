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
#include "nn.h"

Matrix* ReLU (Matrix* mat, long long i, long long j, va_list args){
    bool forward = va_arg(args, int);

    double x = mat_read(mat, i, j);
    if (x < 0) return 0;
    return mat_write(mat, i, j, forward ? x : 1);
}

Matrix* Sigmoid (Matrix* mat, long long i, long long j, va_list args){
    bool forward = va_arg(args, int);
    double x = mat_read(mat, i, j);

    double activation = 1 / (1 + exp(-x));
    if (forward){
        return mat_write(mat, i, j, activation);
    }else{
        return mat_write(mat, i, j, 1 - activation);
    }
}

Layer* nn_buildLayer(long long input, long long output){
    Matrix* weights = xmat_rand(input, output);
    if (weights == NULL){
        printf("Build layer failed: Can't initialize weights.");
        return NULL;
    }

    Layer* layer = malloc(sizeof(Layer));
    if (layer == NULL){
        printf("Build layer failed.");
        return NULL;
    }
    layer->weights = weights;
    return layer;
}

NN* nn_buildNN(
    long long input_size, 
    long long hidden_size, 
    long long ouptut_size, 
    long long hidden_num, 
    MatrixElementOperation activation
    ){
    NN* nn = malloc(sizeof(NN));
    if (nn == NULL){
        fprintf(stderr, "Build NN failed: Can't allocate memory for NN.");
        return NULL;
    }

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->hidden_num = hidden_num;
    nn->layers = calloc(hidden_num + 2, sizeof(Layer*));
    if (nn->layers == NULL){
        fprintf(stderr, "Build NN failed: Can't allocate memory for layers.");
        free(nn);
        return NULL;
    }
    
    // Input Layer
    Layer* input_layer = nn_buildLayer(input_size + 1, hidden_size + 1);
    if (input_layer == NULL){
        fprintf(stderr, "Build NN failed: Can't allocate memory for input layer.");
        free(nn->layers);
        free(nn);
        return NULL;
    }
    
    nn->layers[0] = input_layer;
    
    // Hidden Layer
    for(long long i=1; i < hidden_num + 1; i++){
        Layer* hidden_layer = nn_buildLayer(hidden_size + 1, hidden_size + 1);
        if (hidden_layer == NULL){
            fprintf(stderr, "Build NN failed: Can't allocate memory for hidden layer.");
            return NULL;
        }
        nn->layers[i] = hidden_layer;
    }
    
    // Output Layer
    Layer* output_layer = nn_buildLayer(hidden_size + 1, ouptut_size);
    if (output_layer == NULL){
        fprintf(stderr, "Build NN failed: Can't allocate memory for output layer.");
        return NULL;
    }
    nn->layers[hidden_num + 1] = output_layer;
    nn->activation = activation;
    return nn;
}

void nn_printNN(NN* nn){
    Layer** layers = nn->layers;
    for(long long k=0; k < nn->hidden_num + 2; k++){
        Matrix* weights = layers[k]->weights;
        if (weights == NULL){
            fprintf(stderr, "Print NN failed: Can't print weights.");
            return;
        }
        printf("Layer %lld at %p:\n", k+1, weights);
        mat_print(weights);
        printf("\n");
    }
}

Matrix* nn_forward(NN* nn, double* input, long long input_size){
    double *biased_input = malloc((input_size + 1) * sizeof(double));
    for (long long i=0; i < input_size; i++){
        biased_input[i] = input[i];
    }
    biased_input[input_size] = 1;
    
    Matrix* temp = mat_create(1, input_size + 1, biased_input);
    for (long long layer = 0; layer < nn->hidden_num + 2; layer++){
        Matrix* weights = nn->layers[layer]->weights;
        Matrix* product = mat_multmat(temp, weights);
        // Activation
        product = xmat_traverse(product, nn->activation, true);
        temp = product;
    }
    return mat_transpose(temp);
}

Matrix* nn_backward(NN* nn, Matrix* forward_output, Matrix* target){
    // Target should be in column matrix.
    if (target->col != 1 || forward_output->col != 1){
        fprintf(stderr, "Backward propagation failed: "
                " Target and forward output should be a column matrix.");
        exit(1);
    }
    // Shape should match
    if (forward_output->row != target->row){
        fprintf(stderr, "Backward propagation failed: "
                "Forward output and target column matrices should have the same height.");
        exit(1);
    }

    // Error: Column Vector
    Matrix* total_error = mat_addmat(target, mat_multscal(forward_output, -1));
    
    Matrix* output = forward_output;

    Matrix* cur_gradients = xmat_traverse(total_error, nn->activation, false);



    // From the back most layer to the first layer.
    for (long long i = nn->hidden_num + 1; i >= 0; i--){
        Matrix* weights = nn->layers[i]->weights;

        Matrix* previous_gradients = mat_multmat(weights, cur_gradients);

        printf("Gradient %lld: \n", i);
        mat_print(cur_gradients);

        cur_gradients = previous_gradients;

        // All neurons in a layer.
        // for (long long n = 0; n < weights->row; n++){
        //     Matrix* neuron = xmat_readrow(weights, n);    // Input weights of the first Nueron.
        //     Matrix* gradient = mat_multmat(total_error, gradient);
        // }
    }

    // TODO: Finish the backward propagation logic.
    // This is returned just to avoid compilation error.
    return xmat_diag(nn->hidden_num + 2, nn->hidden_num + 2, 0);
}