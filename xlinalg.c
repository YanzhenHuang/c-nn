/**
 * @file xlinalg.c
 * @author Huang Yanzhen (yanzhenhuangwork@gmail.com)
 * @brief Advanced functions for linear algebra calculations.
 * @version 0.1
 * @date 2024-12-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include "xlinalg.h"

Matrix* xmat_traverse(Matrix* mat, MatrixElementOperation operation, ...){
    va_list args;
    va_start(args, operation);
    for (long long i=0; i < mat->row; i++){
        for (long long j=0; j < mat->col; j++){
            mat = operation(mat, i, j, args);
        }
    }

    va_end(args);
    return mat;
}

Matrix* _set_diagonal(Matrix* mat, long long i, long long j, va_list args){
    double val = va_arg(args, double);
    if (i == j){
        mat_write(mat, i, j, val);
    }
    return mat;
}

Matrix* _set_random(Matrix* mat, long long i, long long j, va_list args){
    double val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    mat = mat_write(mat, i, j, val);
    return mat;
}

Matrix* xmat_diag(long long row, long long col, double val){
    double* empty_data = malloc(row * col * sizeof(double));
    Matrix* diag_mat = mat_create(row, col, empty_data);

    return xmat_traverse(diag_mat, _set_diagonal, val);
}

Matrix* xmat_rand(long long row, long long col){
    srand((unsigned int) time(NULL));

    double* empty_data = malloc(row * col * sizeof(double));
    Matrix* rand_mat = mat_create(row, col, empty_data);

    return xmat_traverse(rand_mat, _set_random);
}

Matrix* xmat_submat(Matrix* mat, long long i_st, long long i_ed, long long j_st, long long j_ed){
    if (i_st > i_ed || j_st > j_ed){
        printf("Acquire sub-matrix failed: Invalid requirement."
                "Starting point shoud always start earlier than ending point.");
        exit(1);
    }
    if (i_st < 0 || i_ed-1 >= mat->row || j_st < 0 || j_ed-1 >= mat->col){
        printf("Acquire sub-matrix failed: Matrix index outof bounds.");
        exit(1);
    }

    double* empty_data = malloc((i_ed - i_st) * (j_ed - j_st) * sizeof(double));
    Matrix* submat = mat_create(i_ed - i_st, j_ed - j_st, empty_data);

    for(long long i=i_st; i < i_ed; i++){
        for(long long j=j_st; j < j_ed; j++){
            double this_val = mat_read(mat, i, j);
            submat = mat_write(submat, i-i_st, j-j_st, this_val);
        }
    }

    return submat;
}

Matrix* xmat_hstack(Matrix* mat_l, Matrix* mat_r){
    if (mat_l->row != mat_r->row){
        printf("Hstack failed: Invalid matrix size. mal_l: %lld x %lld, mat_r: %lld x %lld.",
                mat_l->row, mat_l->col, mat_r->row, mat_r->col);
        exit(1);
    }

    double* empty_data = malloc((mat_l->row) * (mat_l->col + mat_r->col) * sizeof(double));
    Matrix* hstack = mat_create(mat_l->row, mat_l->col + mat_r->col, empty_data);

    // Copy mat_l
    for (long long i = 0; i < mat_l->row; i++) {
        for (long long j = 0; j < mat_l->col; j++) {
            double this_val = mat_read(mat_l, i, j);
            mat_write(hstack, i, j, this_val);
        }
    }

    // Copy mat_r
    for (long long i = 0; i < mat_r->row; i++) {
        for (long long j = 0; j < mat_r->col; j++) {
            double this_val = mat_read(mat_r, i, j);
            mat_write(hstack, i, j + mat_l->col, this_val);
        }
    }

    return hstack;
}

double xmat_det(Matrix*mat){
    if(mat->row != mat->col){
        printf("Calculate determinant failed: Calculate matrix determinant failed: Matrix is not square.");
        exit(1);
    }

    // Recursive method
    if(mat->row == 2){
        // a*d-b*c
        return mat_read(mat, 0, 0) * mat_read(mat, 1, 1) - mat_read(mat, 0, 1) * mat_read(mat, 1, 0);
    }

    double det_val = 0.0;
    int sign = 1;

    // Otherwise, traverse the first row
    for(long long j=0; j < mat->col; j++){
        double anchor = mat_read(mat, 0, j);
        
        Matrix* sub_matrix = (Matrix*) malloc(sizeof(Matrix));

        // Sub-matrix
        if (j==0){
            sub_matrix = xmat_submat(mat, 1, mat->row, 1, mat->col);
        }else if(j == mat->col-1){
            sub_matrix = xmat_submat(mat, 1, mat->row, 0, mat->col-1);
        }else{
            Matrix* submat_l = xmat_submat(mat, 1, mat->row, 0, j);
            Matrix* submat_r = xmat_submat(mat, 1, mat->row, j+1, mat->col);
            sub_matrix = xmat_hstack(submat_l, submat_r);
            free(submat_l);
            free(submat_r);
        }
        
        // Accumulate
        double increment = anchor * sign * xmat_det(sub_matrix);
        sign *= -1;
        det_val += increment;
    }

    return det_val;
}

Matrix* xmat_readrow(Matrix*mat, long long i){
    if (i < 0 || i >= mat->row){
        printf("Read row failed: Matrix index outof bounds.");
        exit(1);
    }

    if (i == -1){
        i = mat->row-1;
    }

    return xmat_submat(mat, i, i+1, 0, mat->col);
}

Matrix* xmat_readcol(Matrix*mat, long long j){
    if ((j < 0 && j!= -1) || j >= mat->col){
        printf("Read column failed: Matrix index outof bounds.");
        exit(1);
    }

    if (j == -1){
        j = mat->col-1;
    }

    return xmat_submat(mat, 0, mat->row, j, j+1);
}

Matrix* xmat_solve(Matrix* A, Matrix* b){
    if (A->col != b->row){
        printf("Solve equation failed: Unable to solve incompatible matrices. \n"
                "A: %lld x %lld, b: %lld x %lld\n", A->row, A->col, b->row, b->col);
        exit(1);
    }

    if (A->col < b->row){
        printf("Solve equation failed: No solution exists.");
        exit(1);
    }
    
    if (A->col > b->row){
        printf("Solve equation failed: Multiple solutions.");
        exit(1);
    }

    // Construct hybrid matrix.
    Matrix* hybrid_mat = xmat_hstack(A, b);

    // Down: Forward Elimination
    for(long long i=0; i < hybrid_mat->row; i++){
        // Check for zero-pivot.
        double pivot = mat_read(hybrid_mat, i, i);
        if (pivot == 0){
            printf("Solve equation failed: Pivot element is zero at row %lld during forward elimination.", i);
            exit(1);
        }

        // Standard row in this iteration.
        // All rows below this row.
        for (long long ib=i+1; ib < hybrid_mat->row; ib++){

            double mult_factor = mat_read(hybrid_mat, ib, i) / mat_read(hybrid_mat, i, i);

            // Col elements in this belowed row.
            for (long long ibj=i; ibj < hybrid_mat->col; ibj++){
                double this_val = mat_read(hybrid_mat, ib, ibj);
                double new_val = this_val - mult_factor * mat_read(hybrid_mat, i, ibj);
                hybrid_mat = mat_write(hybrid_mat, ib, ibj, new_val);
            }
        }
    }

    // Up: Backward Substitution
    for(long long i=hybrid_mat->row-1; i >= 0; i--){
        // Check vor zero-pivot.
        double pivot = mat_read(hybrid_mat, i, i);
        if (pivot == 0){
            printf("Solve equation failed: Pivot element is zero at"
                    " row %lld during backward substitution.", i);
            exit(1);
        }

        // Standard row in this iteration.
        // All rows above this row
        for (long long ia = i-1; ia >= 0; ia--){
            
            double mult_factor = mat_read(hybrid_mat, ia, i) / mat_read(hybrid_mat, i, i);

            // Col elements in this belowed row.
            for (long long iaj=i; iaj < hybrid_mat->col; iaj++){
                double this_val = mat_read(hybrid_mat, ia, iaj);
                double new_val = this_val - mult_factor * mat_read(hybrid_mat, i, iaj);
                hybrid_mat = mat_write(hybrid_mat, ia, iaj, new_val);
            }
        }
    }

    // Pivot Elimination: Divide end values by its pivot.
    for (long long i=0; i < hybrid_mat->row; i++){
        double pivot = mat_read(hybrid_mat, i, i);
        for (long long j=0; j < hybrid_mat->col; j++){
            double this_val = mat_read(hybrid_mat, i, j);
            double new_val = this_val / pivot;
            hybrid_mat = mat_write(hybrid_mat, i, j, new_val);
        }
    }

    // Matrix* x = xmat_readcol(hybrid_mat, -1);

    Matrix* x = xmat_submat(hybrid_mat, 0, hybrid_mat->row, hybrid_mat->col - b->col, hybrid_mat->col);

    return x;
}

Matrix* xmat_inv(Matrix*mat){
    if(mat->row != mat->col){
        printf("Inverse matrix failed: Matrix is not square.");
        exit(1);
    }

    Matrix* identity = xmat_diag(mat->row, mat->col, 1);
    return xmat_solve(mat, identity);
}

bool xmat_isEqual(Matrix* mat_1, Matrix* mat_2){
    if (mat_1->row != mat_2->row ||
        mat_1->col != mat_2->col
    ){
        return false;
    }

    return memcmp(mat_1->data, mat_2->data, mat_1->row * mat_1->col * sizeof(double)) == 0;
}

bool xmat_isSquare(Matrix* mat){
    return mat->row == mat->col;
}

bool xmat_isSymm(Matrix* mat){
    if (!xmat_isSquare(mat)) return false;
    return xmat_isEqual(mat, mat_transpose(mat));
}

bool xmat_isOrth(Matrix* mat){
    Matrix* mat_T = mat_transpose(mat);
    Matrix* eye = xmat_diag(mat->row, mat->col, 1);

    return xmat_isEqual(mat_multmat(mat, mat_T), eye);
}