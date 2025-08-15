/**
 * Author:  TripleZ<me@triplez.cn>
 * Date:    2018-08-17
 */

#include <immintrin.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {
    
    // Single-precision
    int i;
    float* floats = (float*) malloc(8 * 8 * sizeof(float));
    for (i = 0; i < 8; i++) {
        floats[i] = (float)(i) + 1.0;
    }
    
    __m256 float_vec = _mm256_loadu_ps(floats);

    float* flo = (float*) &float_vec;
    printf("float:\t\t%f, %f, %f, %f, %f, %f, %f, %f\n", flo[0], flo[1], flo[2], flo[3], flo[4], flo[5], flo[6], flo[7]);

    // Double-precision
    double* doubles = (double*) malloc(4 * 8 * sizeof(double));
    for (i = 0; i < 4; i++) {
        doubles[i] = (double)(i) + 1.0;
    }

    __m256d double_vec = _mm256_loadu_pd(doubles);

    double* dou = (double*) &double_vec;
    printf("double:\t\t%lf, %lf, %lf, %lf\n", dou[0], dou[1], dou[2], dou[3]);

    // 32-bit Integer
    int* _int = (int*) malloc(8 * 8 * sizeof(int));
    for (i = 0; i < 8; i++) {
        _int[i] = i + 1;
    }

    __m256i int_vec = _mm256_loadu_si256((const __m256i*)_int);

    int* i_v = (int*) &int_vec;
    printf("int:\t\t%d, %d, %d, %d, %d, %d, %d, %d\n", i_v[0], i_v[1], i_v[2], i_v[3], i_v[4], i_v[5], i_v[6], i_v[7]);
    free(floats);
    free(doubles);
    free(_int);
    return 0;
}
