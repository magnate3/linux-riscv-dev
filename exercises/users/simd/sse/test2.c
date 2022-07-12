#include <stdio.h>
#include  <immintrin.h>
int  main()
{
	__m128 v0 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	__m128 v1 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	__m128 result = _mm_add_ps(v0, v1);
	float d[4];
        _mm_storeu_ps(d, result);
        //_mm128_storeu_ps(d, result);
	printf("%f,%f,%f,%f \n", d[0], d[1], d[2], d[3]);
}
