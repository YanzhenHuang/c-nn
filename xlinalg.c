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
#include <math.h>
#include "linalg.h"

Matrix* xmat_submat(Matrix* mat, long long i_st, long long i_ed, long long j_st, long long j_ed){
    if (i_st > i_ed || j_st > j_ed){
        printf("Acquire sub-matrix failed: Invalid requirement. Starting point shoud always start earlier than ending point.");
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
    if (A->row != b->row){
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
    double* hybrid_mat_data = malloc(A->row * (A->col+1) * sizeof(double));
    for (long long i=0; i < A->row; i++){
        // Each row 
        for (long long j=0; j < A->col; j++){
            hybrid_mat_data[i*(A->col+1)+j] = mat_read(A, i, j);
        }
        hybrid_mat_data[i*(A->col+1)+A->col] = mat_read(b, i, 0);
    }

    Matrix* hybrid_mat = mat_create(A->row, A->col+1, hybrid_mat_data);

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
            printf("Solve equation failed: Pivot element is zero at row %lld during backward substitution.", i);
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

    Matrix* x = xmat_readcol(hybrid_mat, -1);

    return x;
}