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

# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <string.h>
# include "nn.h"

int demo_xlinalg(){
        printf("===== Basic Matrix Operations =====\n");
    Matrix* L = mat_create(2,3, (double[]){
        1, 2, 3,
        4, 5, 6
    });
    Matrix* R = mat_create(3,4, (double[]){
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18
    });
    Matrix* Mul = mat_multmat(L, R);
    Matrix* RT = mat_transpose(R);
    Matrix* Add = mat_addmat(L, RT);

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
    Matrix* A = mat_create(3,3, (double[]){
        3, 2, 1, 
        -1, -3, -1, 
        1, -2, -2
    });
    mat_print(A);

    printf("b:\n");
    Matrix* b = mat_create(3,1, (double[]){
        -7, 
        5, 
        4
    });
    mat_print(b);

    printf("Solved that x:\n");
    Matrix* x = xmat_solve(A, b);
    mat_print(x);

    printf("===== Matrix Determinant Calculation =====\n");

    Matrix* C = mat_create(5, 5, (double[]){
        4, 0, -7, 3, -5,
        0, 0, 2, 0, 0,
        7, 3, -6, 4, -8,
        5, 0, 5, 2, -3,
        0, 0, 9, -1, 2
    });

    printf("Matrix C (%lld x %lld):\n", C->row, C->col);
    mat_print(C);

    double detC = xmat_det(C);
    printf("Calculated Determinant: %f\n", detC);
    printf("Matrix acquired from: https://www.youtube.com/watch?v=crCsJy1lKXI\n\n");

    printf("===== Matrix Inverse Calculation =====\n");

    Matrix* D = mat_create(2, 2, (double[]){
        1, 2,
        3, 4
    });

    printf("Matrix D (%lld x %lld):\n", D->row, D->col);
    mat_print(D);

    Matrix* invD = xmat_inv(D);

    printf("Calculated Inverse:\n");
    mat_print(invD);

    printf("===== Simple Matrix Properties =====\n");

    Matrix* Symm = mat_create(3, 3, (double[]){
        1,2,3,
        2,5,4,
        3,4,6
    });

    Matrix* Asym = mat_create(3, 3, (double[]){
        1,2,3,
        7,5,4,
        3,4,6
    });

    Matrix* Orth = mat_create(3, 3, (double[]){
        0, -1, 0,
        1, 0, 0,
        0, 0, -1
    });

    Matrix* Norm = mat_create(3, 3, (double[]){
        1, -1, 0,
        1, 0, 0,
        0, 0, -1
    });

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

void demo_nn(){
    NN* nn = nn_buildNN(2, 3, 2, 5, Sigmoid);
    nn_printNN(nn);
    
    Matrix* output = nn_forward(nn, (double[]){1,2}, 2);
    mat_print(output);
}

int main(int argc, char* argv[], char**envp){
    if (argc < 2){
        fprintf(stderr, "Usage: %s <demo>\n", argv[0]);
        exit(1);
    }
    // Parse terminal arguments.
    char* op = argv[1];
    char* val = argv[2];

    if (strcmp(op, "-demo") != 0){
        fprintf(stderr, "Unknown argument %s", argv[1]);
    }
    if (strcmp(val, "xlinalg") == 0){
        demo_xlinalg();
    }else if(strcmp(val, "nn")){
        demo_nn();
    }else{
        fprintf(stderr, "Unknown demo type %s", val);
        exit(1);
    }

    return 0;
}