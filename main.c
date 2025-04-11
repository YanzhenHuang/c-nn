/**
 * @file main.c
 * @author Huang Yanzhen (yanzhenhuangwork@gmail.com)
 * @brief Main demo function that you can alter to explore this project.
 * @version 0.1
 * @date 2024-12-25
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "linalg.h"
#include "xlinalg.h"
#include "nn.h"

int demo_xlinalg()
{
    printf("===== Basic Matrix Operations =====\n");
    Matrix *L = mat_create(2, 3, (double[]){1, 2, 3, 4, 5, 6});
    Matrix *R = mat_create(3, 4, (double[]){7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    Matrix *Mul = mat_multmat(L, R);
    Matrix *RT = mat_transpose(R);
    Matrix *Add = mat_addmat(L, RT);

    printf("Left Matrix (L):\n");
    mat_print(L);

    printf("Right Matrix (R):\n");
    mat_print(R);

    printf("Multiplied Matrix (M=LR):\n");
    mat_print(Mul);

    printf("Added Matrix (A=L+RT):\n");
    mat_print(Add);

    printf("===== Basic Matrix Equation Solving =====\n");
    printf("Solving Ax=b. Where:\n A:\n");
    Matrix *A = mat_create(3, 3, (double[]){3, 2, 1, -1, -3, -1, 1, -2, -2});
    mat_print(A);

    printf("b:\n");
    Matrix *b = mat_create(3, 1, (double[]){-7, 5, 4});
    mat_print(b);

    printf("Solved that x:\n");
    Matrix *x = xmat_solve(A, b);
    mat_print(x);

    printf("===== Matrix Determinant Calculation =====\n");

    Matrix *C = mat_create(5, 5, (double[]){4, 0, -7, 3, -5, 0, 0, 2, 0, 0, 7, 3, -6, 4, -8, 5, 0, 5, 2, -3, 0, 0, 9, -1, 2});

    printf("Matrix C (%lld x %lld):\n", C->row, C->col);
    mat_print(C);

    printf("Element Sum of Matrix C: %llf\n", mat_elemSum(C));

    double detC = xmat_det(C);
    printf("Calculated Determinant: %f\n", detC);
    printf("Matrix acquired from: https://www.youtube.com/watch?v=crCsJy1lKXI\n\n");

    printf("===== Matrix Inverse Calculation =====\n");

    Matrix *D = mat_create(2, 2, (double[]){1, 2, 3, 4});

    printf("Matrix D (%lld x %lld):\n", D->row, D->col);
    mat_print(D);

    Matrix *invD = xmat_inv(D);

    printf("Calculated Inverse:\n");
    mat_print(invD);

    printf("===== Simple Matrix Properties =====\n");

    Matrix *Symm = mat_create(3, 3, (double[]){1, 2, 3, 2, 5, 4, 3, 4, 6});

    Matrix *Asym = mat_create(3, 3, (double[]){1, 2, 3, 7, 5, 4, 3, 4, 6});

    Matrix *Orth = mat_create(3, 3, (double[]){0, -1, 0, 1, 0, 0, 0, 0, -1});

    Matrix *Norm = mat_create(3, 3, (double[]){1, -1, 0, 1, 0, 0, 0, 0, -1});

    printf("Symmetric Matrix (%lld x %lld):\n", Symm->row, Symm->col);
    mat_print(Symm);

    bool identify_Symm = xmat_isSymm(Symm);
    printf("Identify as: %d\n", identify_Symm);

    printf("Asymmetric Matrix (%lld x %lld):\n", Asym->row, Asym->col);
    mat_print(Asym);

    bool identify_Asym = xmat_isSymm(Asym);
    printf("Identify as: %d\n", identify_Asym);

    printf("Orthogonal Matrix (%lld x %lld):\n", Orth->row, Orth->col);
    mat_print(Orth);

    bool identify_Orth = xmat_isOrth(Orth);
    printf("Identify as: %d\n", identify_Orth);

    printf("Un-orthogonal Matrix (%lld x %lld):\n", Norm->row, Norm->col);
    mat_print(Norm);

    bool identify_Norm = xmat_isOrth(Norm);
    printf("Identify as: %d\n", identify_Norm);

    return 0;
}

void demo_nn()
{
    NN *nn = nn_buildNN(2, 3, 2, 1, ReLU, nngrad_CELoss);
    printf("Initial weights.\n");
    nn_printNN(nn);

    Matrix *output = nn_forward(nn, (double[]){1, 2}, 2);
    printf("Output of forward.\n");
    mat_print(output);

    Matrix *Yd = mat_create(2, 1, (double[]){2, 1});
    nn = nn_backward(nn, output, Yd, 1e-2);
    printf("Trained weights.\n");
    nn_printNN(nn);
}

void demo_xornn()
{
    srand(time(0));

    // A simple 2-layered NN to calculate the XOR problem.
    NN *xor_nn = nn_buildNN(2, 2, 1, 1, ReLU, nngrad_CELoss);

    printf("Weights before training....\n\n");
    nn_printNN(xor_nn);

    int x_1[2] = {0, 1};
    int x_2[2] = {1, 0};

    for (int epoch = 0; epoch < 10; epoch++)
    {
        // printf("Epoch %d\n", epoch + 1);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                int xor = x_1[i] ^ x_2[j];
                Matrix *output = nn_forward(xor_nn, (double[]){x_1[i], x_2[j]}, 2);
                Matrix *Yd = mat_create(1, 1, (double[]){xor});
                nn_backward(xor_nn, Yd, output, 1e-2);
            }
        }
    }

    printf("Weights after training....\n\n");
    nn_printNN(xor_nn);

    double *results = malloc(9999 * sizeof(double));

    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;

    printf("\nEvaluating Samples...\n\n");
    for (int epoch = 0; epoch < 5; epoch++)
    {
        // printf("Sample %d\n", epoch+1);
        int x_1 = rand() % 2;
        int x_2 = rand() % 2;
        int xor = x_1 ^ x_2;

        Matrix *output = nn_forward(xor_nn, (double[]){x_1, x_2}, 2);
        Matrix *Yd = mat_create(1, 1, (double[]){xor});
        int res = mat_read(output, 0, 0) > 0.5 ? 1 : 0;

        results[epoch] = mat_read(output, 0, 0);

        if (res == 1 && xor == 1)
        {
            tp++;
        }
        else if (res == 1 && xor == 0)
        {
            fp++;
        }
        else if (res == 0 && xor == 1)
        {
            fn++;
        }
        else if (res == 0 && xor == 0)
        {
            tn++;
        }
    }

    double precision = tp / (double)(tp + fp);
    double recall = tp / (double)(tp + fn);
    double accuracy = (tp + tn) / (double)(tp + tn + fp + fn);
    double f1 = (2 * precision * recall) / (double)(precision + recall);

    printf("\n~~~ NN Final output states ~~~\n");
    mat_print(xor_nn->output_states[2]);

    printf("\n~~~ Confusion Matrix ~~~\n");
    printf("TP: %f, TN: %f, FP: %f, FN: %f\n", tp, tn, fp, fn);

    printf("\n~~~ Evaluation ~~~\n");
    printf("Prc: %f\n", precision);
    printf("Rec: %f\n", recall);
    printf("Acc: %f\n", accuracy);
    printf("F1: %f\n", f1);

    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%d, %f;  ", i, results[i]);
    // }

    free(xor_nn);
}

int main(int argc, char *argv[], char **envp)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <demo>\n", argv[0]);
        exit(1);
    }
    // Parse terminal arguments.
    char *op = argv[1];
    char *val = argv[2];

    if (strcmp(op, "-demo") != 0)
    {
        fprintf(stderr, "Unknown argument %s", argv[1]);
    }
    if (strcmp(val, "xlinalg") == 0)
    {
        demo_xlinalg();
    }
    else if (strcmp(val, "nn") == 0)
    {
        demo_nn();
    }
    else if (strcmp(val, "xornn") == 0)
    {
        demo_xornn();
    }
    else
    {
        fprintf(stderr, "Unknown demo type %s", val);
        exit(1);
    }

    return 0;
}