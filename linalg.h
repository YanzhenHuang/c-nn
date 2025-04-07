#ifndef LINALG_H
#define LINALG_H

/**
 * @brief Matrix struct.
 *
 */
typedef struct
{
    long long row;
    long long col;
    double *data;
} Matrix;

/**
 * @brief Create a matrix with given size and data.
 *
 * @param row Row size of matrix.
 * @param col Column size of matrix.
 * @param data Data array of the matrix.
 * @return Matrix*
 */
Matrix *mat_create(long long row, long long col, double *data);

/**
 * @brief Read a matrix value.
 *
 * @param matrix Matrix struct pointer.
 * @param i Row index of the value.
 * @param j Column index of the value.
 * @return double
 */
double mat_read(Matrix *matrix, long long i, long long j);

/**
 * @brief Write a value into a matrix.
 *
 * @param matrix Matrix struct pointer.
 * @param i Row index of the value.
 * @param j Column index of the value.
 * @param val The intended value to be written.
 * @return Matrix*
 */
Matrix *mat_write(Matrix *matrix, long long i, long long j, double val);

/**
 * @brief Print the matrix in the console.
 *
 * @param matrix Matrix struct pointer.
 */
void mat_print(Matrix *matrix);

/**
 * @brief Element sum of a matrix.
 *
 * @param matrix Matrix struct pointer.
 * @return double
 */
double mat_elemSum(Matrix *matrix);

/**
 * @brief Transpose a matrix.
 *
 * @param matrix Matrix struct pointer.
 * @return Matrix*
 */
Matrix *mat_transpose(Matrix *matrix);

/**
 * @brief Matrix addition with scalar.
 *
 * @param mat Matrix struct pointer of the matrix
 * @param val Scalar value.
 * @return Matrix*
 */
Matrix *mat_addscal(Matrix *mat, double val);

/**
 * @brief Matrix multiplication with scalar.
 *
 * @param mat Matrix struct pointer of the matrix.
 * @param val Scalar value.
 * @return Matrix*
 */
Matrix *mat_multscal(Matrix *mat, double val);

/**
 * @brief Matrix addition.
 *
 * @param mat_1 Matrix struct pointer of the first matrix.
 * @param mat_2 Matrix struct pointer of the second matrix.
 * @return Matrix*
 */
Matrix *mat_addmat(Matrix *mat_1, Matrix *mat_2);

/**
 * @brief Matrix subtraction.
 *
 * @param mat_1 Matrix struct pointer of the first matrix.
 * @param mat_2 Matrix struct pointer of the first matrix.
 * @return Matrix*
 */
Matrix *mat_difmat(Matrix *mat_1, Matrix *mat_2);

/**
 * @brief Point-wise production of two matrices.
 *
 * @param mat_1 Matrix struct pointer of the first matrix.
 * @param mat_2 Matrix struct pointer of the second matrix.
 * @return Matrix*
 */
Matrix *mat_pwpmat(Matrix *mat_1, Matrix *mat_2);

/**
 * @brief Matrix multiplication.
 *
 * @param mat_l Left matrix.
 * @param mat_r Right matrix.
 * @return Matrix*
 */
Matrix *mat_multmat(Matrix *mat_l, Matrix *mat_r);

#endif
