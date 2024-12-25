# include <stdio.h>
# include <stdlib.h>
# include "xlinalg.h"

int main(){
    // Matrix* L = mat_create(2,3, (double[]){1,2,3,4,5,6});
    // Matrix* R = mat_create(3,2, (double[]){7,8,9,10,11,12});
    // Matrix* M = mat_multmat(L, R);

    // Matrix* RT = mat_transpose(R);

    // Matrix* A = mat_addmat(L, RT);

    // mat_print(L);
    // mat_print(R);
    // mat_print(M);
    // mat_print(A);

    Matrix* A = mat_create(3,3, (double[]){3, 2, 1, -1, -3, -1, 1, -2, -2});
    Matrix* b = mat_create(3,1, (double[]){-7, 5, 4});
    Matrix* hybrid = xmat_solve(A, b);

    mat_print(hybrid);

    // double* data = malloc(10 * 10 * sizeof(double));

    // for (int i=0; i<100; i++){
    //     data[i]= (double) i;
    // }

    // Matrix* Big = mat_create(10, 10, data);
    // Matrix* sub = xmat_submat(Big, 0, 10, 5, 7);

    // mat_print(Big);
    // mat_print(sub);

    return 0;
}