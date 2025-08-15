#include <stdio.h>
#include  <immintrin.h>
#include <emmintrin.h>
__m256i a;
int  main()
{
     a=_mm256_set_epi32(1,2,3,4,5,6,7,8);
     printf("%llx \n",a[2]);
     return 0;
}
