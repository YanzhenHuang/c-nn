#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "linalg.h"

typedef struct {
    Matrix* matrix;
    bool is_null;
} Optional;

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
Matrix* xmat_submat(Matrix* mat, long long i_st, long long i_ed, long long j_st, long long j_ed);

/**
 * @brief Acquire a row of a matrix.
 * 
 * @param mat Mother matrix struct pointer.
 * @param i Row index
 * @return Matrix* 
 */
Matrix* xmat_readrow(Matrix*mat, long long i);

/**
 * @brief Acquire a column of a matrix.
 * 
 * @param mat Mother matrix struct pointer.
 * @param j Column index.
 * @return Matrix* 
 */
Matrix* xmat_readcol(Matrix*mat, long long j);

/**
 * @brief Calculate the inverse of a matrix.
 * 
 * @param mat Matrix struct pointer.
 * @return Matrix* 
 */
Matrix* xmat_inv(Matrix*mat);

/**
 * @brief Calculate the determinant of a matrix.
 * 
 * @param mat Matrix struct pointer.
 * @return Matrix* 
 */
double xmat_det(Matrix*mat);

/**
 * @brief Solve for matrix equation Ax=b.
 * 
 * @param A A matrix
 * @param b b vector
 * @return Matrix* 
 */
Matrix* xmat_solve(Matrix* A, Matrix* b);