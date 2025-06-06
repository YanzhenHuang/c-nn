/**
 * @file linalg.c
 * @author Huang Yanzhen (yanzhenhuangwork@gmail.com)
 * @brief Basic functions for linear algebra calculations.
 * @version 0.1
 * @date 2024-12-25
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"

Matrix *mat_create(long long row, long long col, double *data)
{
    if (row <= 0 || col <= 0)
    {
        fprintf(stderr, "Matrix Create Failed: Invalid matrix size\n");
        exit(1);
    }

    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    matrix->row = row;
    matrix->col = col;
    matrix->data = data;

    return matrix;
}

Matrix *mat_copy(Matrix *matrix)
{
    Matrix *newMatrix = mat_create(matrix->row, matrix->col, matrix->data);
    return newMatrix;
}

double mat_read(Matrix *matrix, long long i, long long j)
{
    if (i < 0 || j < 0 || i >= matrix->row || j >= matrix->col)
    {
        fprintf(stderr, "Matrix Read Failed: Matrix location index out of bounds.\n");
        exit(1);
    }
    return matrix->data[i * matrix->col + j];
}

void mat_write(Matrix *matrix, long long i, long long j, double val)
{
    if (i < 0 || j < 0 || i >= matrix->row || j >= matrix->col)
    {
        fprintf(stderr, "Matrix Write Failed: Matrix location index out of bounds.\n");
        exit(1);
    }
    matrix->data[i * matrix->col + j] = val;
    // return matrix;
}

void mat_print(Matrix *matrix)
{
    if (matrix->row <= 0 || matrix->col <= 0)
    {
        fprintf(stderr, "Matrix Print Failed: Malicious matrix size.");
        exit(1);
    }
    for (long long i = 0; i < matrix->row; i++)
    {
        for (long long j = 0; j < matrix->col; j++)
        {
            printf("%.4f  ", mat_read(matrix, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

double mat_elemSum(Matrix *matrix)
{
    if (matrix->row <= 0 || matrix->col <= 0)
    {
        fprintf(stderr, "Matrix Element-wise Sum Failed: Malicious matrix size.");
        exit(1);
    }
    double sum = 0;
    for (long long i = 0; i < matrix->row * matrix->col; i++)
    {
        sum += matrix->data[i];
    }

    return sum;
}

Matrix *mat_transpose(Matrix *matrix)
{
    if (matrix->row <= 0 || matrix->col <= 0)
    {
        fprintf(stderr, "Matrix Transpose Sum Failed: Malicious matrix size.");
        exit(1);
    }

    double *empty_data = malloc(matrix->row * matrix->col * sizeof(double));

    Matrix *transposed = mat_create(matrix->col, matrix->row, empty_data);

    for (long long i = 0; i < matrix->row; i++)
    {
        for (long long j = 0; j < matrix->col; j++)
        {
            mat_write(transposed, j, i, mat_read(matrix, i, j));
        }
    }

    return transposed;
}

Matrix *mat_addscal(Matrix *mat, double val)
{

    if (mat->row <= 0 || mat->col <= 0)
    {
        fprintf(stderr, "Matrix Add Scalar Failed: Malicious matrix size.");
        exit(1);
    }
    double *added_data = malloc(mat->row * mat->col * sizeof(double));

    for (long long i = 0; i < mat->row * mat->col; i++)
    {
        added_data[i] = mat->data[i] + val;
    }

    Matrix *added = mat_create(mat->row, mat->col, added_data);

    return added;
}

Matrix *mat_multscal(Matrix *mat, double val)
{
    if (mat->row <= 0 || mat->col <= 0)
    {
        fprintf(stderr, "Matrix Multiply Scalar Failed: Malicious matrix size.");
        exit(1);
    }
    double *multiplied_data = malloc(mat->row * mat->col * sizeof(double));

    for (long long i = 0; i < mat->row * mat->col; i++)
    {
        multiplied_data[i] = mat->data[i] * val;
    }

    Matrix *multiplied = mat_create(mat->row, mat->col, multiplied_data);
    return multiplied;
}

Matrix *mat_addmat(Matrix *mat_1, Matrix *mat_2)
{
    if (mat_1->row <= 0 || mat_1->col <= 0 || mat_2->row <= 0 || mat_2->col <= 0)
    {
        fprintf(stderr, "Matrix Add Matrix Failed: Malicious matrix size of mat_1.");
        exit(1);
    }

    if (mat_1->row != mat_2->row || mat_1->col != mat_2->col)
    {
        fprintf(stderr,
                "Matrix Add Matrix Failed:"
                "Cannot add matrix with different size.\n"
                "mat_1 have size %lld x %lld while mat_2 have size %lld x %lld.",
                mat_1->row, mat_1->col, mat_2->row, mat_2->col);
    }

    double *added_data = malloc(mat_1->row * mat_1->col * sizeof(double));

    for (long long i = 0; i < mat_1->row * mat_1->col; i++)
    {
        added_data[i] = mat_1->data[i] + mat_2->data[i];
    }

    Matrix *added = mat_create(mat_1->row, mat_1->col, added_data);

    return added;
}

Matrix *mat_difmat(Matrix *mat_1, Matrix *mat_2)
{
    return mat_addmat(mat_1, mat_multscal(mat_2, -1));
}

Matrix *mat_pwpmat(Matrix *mat_1, Matrix *mat_2)
{
    if (mat_1->row <= 0 || mat_1->col <= 0 || mat_2->row <= 0 || mat_2->col <= 0)
    {
        fprintf(stderr, "Matrix Point-wise Multiply Matrix Failed: Malicious matrix size of mat_1.");
        exit(1);
    }

    if (mat_1->row != mat_2->row || mat_1->col != mat_2->col)
    {
        fprintf(stderr,
                "Matrix Point-wise Multiply Matrix Failed:"
                "Cannot point-wise product matrix with different size.\n"
                "mat have size %lld x %lld while val have size %lld x %lld.",
                mat_1->row, mat_1->col, mat_2->row, mat_2->col);
        exit(1);
    }

    double *multiplied_data = malloc(mat_1->row * mat_1->col * sizeof(double));

    for (long long i = 0; i < mat_1->row * mat_1->col; i++)
    {
        multiplied_data[i] = mat_1->data[i] * mat_2->data[i];
    }

    Matrix *added = mat_create(mat_1->row, mat_1->col, multiplied_data);

    return added;
}

Matrix *mat_multmat(Matrix *mat_l, Matrix *mat_r)
{
    if (mat_l->row <= 0 || mat_l->col <= 0 || mat_r->row <= 0 || mat_r->col <= 0)
    {
        fprintf(stderr, "Matrix Multiply Matrix Failed: Malicious matrix size of mat_1.");
        exit(1);
    }

    if (mat_l->col != mat_r->row)
    {
        fprintf(stderr,
                "Matrix Multiply Matrix Failed:"
                "Cannot multiply matrx with size %lld x %lld and size %lld x %lld.",
                mat_l->row, mat_l->col, mat_r->row, mat_r->col);
        printf("\nTrying to multiply:\n");
        mat_print(mat_l);
        mat_print(mat_r);
        exit(1);
    }

    double *empty_data = malloc(mat_l->row * mat_r->col * sizeof(double));
    Matrix *multiplied = mat_create(mat_l->row, mat_r->col, empty_data);

    for (long long i = 0; i < multiplied->row; i++)
    {
        for (long long j = 0; j < multiplied->col; j++)
        {
            // Locate the row & col of the new matrix.
            double lin_comb = 0;
            for (long long k = 0; k < mat_l->col; k++)
            {
                lin_comb += mat_read(mat_l, i, k) * mat_read(mat_r, k, j);
            }
            mat_write(multiplied, i, j, lin_comb);
        }
    }

    return multiplied;
}
