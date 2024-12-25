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
# include "xlinalg.h"

int main(){
    printf("===== Basic Matrix Operations ===\n");
    Matrix* L = mat_create(2,3, (double[]){
        1, 2, 3, 
        4, 5, 6
    });
    Matrix* R = mat_create(3,2, (double[]){
        7, 8, 
        9, 10,
        11, 12
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

    printf("===== Basic Matrix Equation Solving ===\n");
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

    printf("===== Matrix Determinant Calculation ===\n");

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
    printf("Matrix acquired from: https://www.youtube.com/watch?v=crCsJy1lKXI");

    return 0;
}