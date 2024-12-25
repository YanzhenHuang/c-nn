# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "linalg.h"

Matrix *mat_create(long long row, long long col, double *data){
    if (row <= 0 || col <= 0){
        printf("Invalid matrix size\n");
        exit(1);
    }

    double last_ele = data[row * col - 1];      // If the data is insufficient an auto hault will happen.

    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->row = row;
    matrix->col = col;
    matrix->data = data;

    return matrix;
}

double mat_read(Matrix *matrix, long long i, long long j){
    if (i >= matrix->row || j >= matrix->col){
        printf("Matrix location index out of bounds.\n");
        exit(1);
    }
    return matrix->data[i * matrix->col + j];
}

Matrix* mat_write(Matrix *matrix, long long i, long long j, double val){
    if (i >= matrix->row || j >= matrix->col){
        printf("Matrix location index out of bounds.\n");
        exit(1);
    }
    matrix->data[i * matrix->col + j] = val;
    return matrix;
}

void mat_print(Matrix* matrix){
    for (long long i=0; i < matrix->row; i++){
        for (long long j=0; j < matrix->col; j++){
            printf("%f ", mat_read(matrix, i,j));
        }
        printf("\n");
    }
    printf("\n");
}

Matrix* mat_transpose(Matrix* matrix){

    double* empty_data = malloc(matrix->row * matrix->col * sizeof(double));

    Matrix* transposed = mat_create(matrix->col, matrix->row, empty_data);

    for (long long i=0; i< matrix->row; i++){
        for (long long j=0; j< matrix->col; j++){
            transposed = mat_write(transposed, j, i, mat_read(matrix, i, j));
        }
    }

    return transposed;
}

Matrix* mat_addscal(Matrix* mat, double val){
    double* empty_data = malloc(mat->row * mat->col * sizeof(double));
    Matrix* added = mat_create(mat->row, mat->col, empty_data);
    for (long long i=0; i<mat->row; i++){
        for (long long j=0; j<mat->col; j++){
            added = mat_write(added, i, j, mat_read(mat, i, j) + val);
        }
    }
    return added;
}

Matrix* mat_multscal(Matrix* mat, double val){
    double* empty_data = malloc(mat->row * mat->col * sizeof(double));
    Matrix* multiplied = mat_create(mat->row, mat->col, empty_data);
    for (long long i=0; i<mat->row; i++){
        for (long long j=0; j<mat->col; j++){
            multiplied = mat_write(multiplied, i, j, mat_read(mat, i, j) * val);
        }
    }
    return multiplied;
}

Matrix* mat_addmat(Matrix* mat_1, Matrix* mat_2){
    if(mat_1->row != mat_2->row || mat_1->col != mat_2->col){
        printf("Cannot add matrix with different size.\n"
                "mat_1 have size %lld x %lld while mat_2 have size %lld x %lld.",
                mat_1->row, mat_1->col, mat_2->row, mat_2->col);
    }

    double* empty_data = malloc(mat_1->row * mat_1->col * sizeof(double));
    Matrix* added = mat_create(mat_1->row, mat_1->col, empty_data);
    for (long long i=0; i<added->row; i++){
        for (long long j=0; j<added->col; j++){
            double added_val = mat_read(mat_1, i, j) + mat_read(mat_2, i, j);
            added = mat_write(added, i, j, added_val);
        }
    }

    return added;
}

Matrix* mat_multmat(Matrix* mat_l, Matrix* mat_r){
    if (mat_l->col != mat_r->row){
        printf("Cannot multiply matrx with size %lld x %lld and size %lld x %lld.", mat_l->row, mat_l->col, mat_r->row, mat_r->col);
        exit(1);
    }

    double* empty_data = malloc(mat_l->row * mat_r->col * sizeof(double));
    Matrix* multiplied = mat_create(mat_l->row, mat_r->col, empty_data);

    for (long long i=0; i<multiplied->row; i++){
        for (long long j=0; j<multiplied->col; j++){
            // Locate the row & col of the new matrix.
            double lin_comb = 0;
            for (long long k=0; k<mat_l->col; k++){
                lin_comb += mat_read(mat_l, i, k) * mat_read(mat_r, k, j);
            }
            multiplied = mat_write(multiplied, i, j, lin_comb);
        }
    }

    return multiplied;
}