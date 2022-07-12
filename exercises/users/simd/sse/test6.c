#include <immintrin.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {

	    // 32-bit integer addition (AVX2)
	    //__m256i epi32_vec_0 = _mm256_set_epi32(8,9,10,11,12,13,14,15);
	    //__m256i epi32_vec_1 = _mm256_set_epi32(17,18,19,20,21,22,23,24);
	    __m256i epi32_vec_0 = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
	    __m256i epi32_vec_1 = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
	    __m256i epi32_result = _mm256_add_epi32(epi32_vec_0, epi32_vec_1);
	    int* i = (int*) &epi32_result;
	    printf("int:\t\t%d, %d, %d, %d, %d, %d, %d, %d\n", i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
	    
	    // 64-bit integer addition (AVX2)
	    __m256i epi64_vec_0 = _mm256_set1_epi64x(8);
	    __m256i epi64_vec_1 = _mm256_set1_epi64x(17);
	    __m256i epi64_result = _mm256_add_epi64(epi64_vec_0, epi64_vec_1);
	    long long int* lo = (long long int*) &epi64_result;
	    printf("long long:\t%lld, %lld, %lld, %lld\n", lo[0], lo[1], lo[2], lo[3]);
    return 0;
}
