#ifndef XLINALG_H
#define XLINALG_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include "linalg.h"

typedef struct
{
    Matrix *matrix;
    bool is_null;
} Optional;

/**
 * @brief Matrix element operation function. Operates on a single matrix element.
 *
 */
typedef Matrix *(*MatrixElementOperation)(Matrix *, long long, long long, va_list);

/**
 * @brief Pointwise operation on two matrices.
 *
 */
typedef Matrix *(*MatrixPointwiseOperation)(Matrix *, Matrix *);

/**
 * @brief Traverse a matrix and operate on single element.
 *
 * @param mat Matrix struct pointer.
 * @param row Matrix height, number of rows.
 * @param col Matrix width, number of columns.
 * @param operation Operation function pointer.
 * @param ... Dynamic arguments.
 * @return Matrix*
 */
Matrix *xmat_traverse(Matrix *mat, MatrixElementOperation operation, ...);

/**
 * @brief Generate a diagonal matrix.
 *
 * @param row Matrix height, number of rows.
 * @param col Matrix width, number of columns.
 * @param val Value of the diagonal.
 * @return Matrix*
 */
Matrix *xmat_diag(long long row, long long col, double val);

/**
 * @brief Generate a square identity matrix of a given size.
 * 
 * @param size The size of the identity square matrix.
 * @return Matrix* 
 */
Matrix *xmat_identity(long long size);

/**
 * @brief Generate a random matrix.
 *
 * @param row Matrix height, number of rows.
 * @param col Matrix width, number of columns.
 * @return Matrix*
 */
Matrix *xmat_rand(long long row, long long col);

/**
 * @brief Acquire a sub-matrix from a mother matrix.
 *
 * @param mat Mother matrix struct pointer.
 * @param i_st Start row index.
 * @param i_ed End row index.
 * @param j_st Start column index.
 * @param j_ed End column index.
 * @return Matrix*
 */
Matrix *xmat_submat(Matrix *mat, long long i_st, long long i_ed, long long j_st, long long j_ed);

/**
 * @brief Stack a left and right matrix horizontally.
 *
 * @param mat_l Left matrix struct pointer.
 * @param mat_r Right matrix struct pointer.
 * @return Matrix*
 */
Matrix *xmat_hstack(Matrix *mat_l, Matrix *mat_r);

/**
 * @brief Horizontally repeat a matrix.
 *
 * @param mat
 * @param n
 * @return Matrix*
 */
Matrix *xmat_hrepeat(Matrix *mat, int n);

/**
 * @brief Stack an up and down matrix vertically.
 *
 * @param mat_u Up matrix struct pointer.
 * @param mat_d Down matrix struct pointer.
 * @return Matrix*
 */
Matrix *xmat_vstack(Matrix *mat_u, Matrix *mat_d);

/**
 * @brief Vertically repeate a matrix.
 *
 * @param mat
 * @param n
 * @return Matrix*
 */

Matrix *xmat_vrepeat(Matrix *mat, int n);

/**
 * @brief Calculate the determinant of a matrix.
 *
 * @param mat Matrix struct pointer.
 * @return Matrix*
 */
double xmat_det(Matrix *mat);

/**
 * @brief Acquire a row of a matrix.
 *
 * @param mat Mother matrix struct pointer.
 * @param i Row index, which row to get.
 * @return Matrix*
 */
Matrix *xmat_readrow(Matrix *mat, long long i);

/**
 * @brief Acquire a column of a matrix.
 *
 * @param mat Mother matrix struct pointer.
 * @param j Column index, which column to get.
 * @return Matrix*
 */
Matrix *xmat_readcol(Matrix *mat, long long j);

/**
 * @brief Solve for matrix equation Ax=b.
 *
 * @param A A matrix struct pointer.
 * @param b b matrix struct pointer.
 * @return Matrix*
 */
Matrix *xmat_solve(Matrix *A, Matrix *b);

/**
 * @brief Calculate the inverse of a matrix.
 *
 * @param mat Matrix struct pointer.
 * @return Matrix*
 */
Matrix *xmat_inv(Matrix *mat);

/**
 * @brief Identify if two matrices are equal.
 *
 * @param mat_1 First matrix struct pointer.
 * @param mat_2 Second matrix struct pointer.
 * @return bool
 */
bool xmat_isEqual(Matrix *mat_1, Matrix *mat_2);

/**
 * @brief Identify if a matrix is a row matrix.
 *
 * @param matrix Matrix struct pointer.
 * @return int*
 */
bool xmat_isRow(Matrix *matrix);

/**
 * @brief Identify if a matrix is a column matrix.
 *
 * @param matrix Matrix struct pointer.
 * @return int*
 */
bool xmat_isCol(Matrix *matrix);

/**
 * @brief Identify if a matrix is a square matrix.
 *
 * @param mat Matrix struct pointer.
 * @return bool
 */
bool xmat_isSquare(Matrix *mat);

/**
 * @brief Identify if a matrix is symmetric.
 *
 * @param mat Matrix struct pointer.
 * @return bool
 */
bool xmat_isSymm(Matrix *mat);

/**
 * @brief Identify if a matrix is orthogonal.
 *
 * @param mat Matrix struct pointer.
 * @return bool
 */
bool xmat_isOrth(Matrix *mat);

/**
 * @brief Identify if a matrix is all 0;
 * 
 * @param mat Matrix struct pointer.
 * @return bool
 */
bool xmat_isZero(Matrix *mat);

/**
 * @brief Element mean of a matrix.
 * 
 * @param mat Matrix struct pointer.
 * @return long long
 */
long long xmat_mean(Matrix *mat);

/**
 * @brief Element standard deviation of a matrix.
 * 
 * @param mat Matrix struct pointer.
 * @return long long 
 */
long long xmat_std(Matrix *mat);

#endif