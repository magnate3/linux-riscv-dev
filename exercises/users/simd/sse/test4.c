#include <immintrin.h>
#include <stdio.h>
#define ARRAY_LENGTH 8
int main(int argc, char* argv[]) {
    __m256i first = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
    __m256i second = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
    __m256i result = _mm256_add_epi32(first, second);
    int* values = (int*) &result;
    printf("int:\t\t%d, %d, %d, %d, %d, %d, %d, %d\n", values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    printf("int:\t\t%d, %d, %d, %d, %d, %d, %d, %d\n", values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    for(int i = 0; i < ARRAY_LENGTH; ++i)
    printf("int:%d, \t\t%d\n", i , values[i]);
    return 0;
}
