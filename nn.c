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
#include <stdlib.h>
#include <math.h>
#include "nn.h"

double ReLU (double x){
    return x > 0 ? x : 0;
}

double Sigmoid (double x){
    return 1 / (1 + exp(-x));
}

Layer* nn_buildLayer(long long input, long long output){
    Matrix* weights = xmat_rand(input, output);
    Layer* layer = malloc(sizeof(Layer));
    layer->weights = weights;
    return layer;
}

NN* nn_buildNN(long long input_size, long long hidden_size, long long ouptut_size, long long hidden_num, int activation){
    NN* nn = malloc(sizeof(NN));
    nn->layers = malloc((hidden_num + 2) * sizeof(Layer*));

    // Input Layer
    Layer* input_layer = nn_buildLayer(input_size + 1, hidden_size + 1);
    nn->layers[0] = input_layer;
    
    // Hidden Layer
    for(long long i=0; i < hidden_num; i++){
        nn->layers[i] = nn_buildLayer(hidden_size + 1, hidden_size + 1);
    }

    // Output Layer
    Layer* output_layer = nn_buildLayer(hidden_size + 1, ouptut_size + 1);
    nn->layers[hidden_num + 1] = output_layer;

    nn->activation = activation;

    return nn;
}