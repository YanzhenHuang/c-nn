# include <stdio.h>
# include <stdlib.h>
# include "linalg.h"

int main(){
    Matrix* L = mat_create(2,3, (double[]){1,2,3,4,5,6});
    Matrix* R = mat_create(3,2, (double[]){7,8,9,10,11,12});
    Matrix* M = mat_multmat(L, R);

    Matrix* RT = mat_transpose(R);

    Matrix* A = mat_addmat(L, RT);

    mat_print(L);
    mat_print(R);
    mat_print(M);
    mat_print(A);

    return 0;
}