#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define WIDTH (1024)
#define ITERATIONS (10)
#define TILE_SIZE (256)

int main() {
    float **P, **M, **N;
    int i, j, k, it;
	int a, b, c;
    struct timeval start, end;
    float totaltime = 0;

    P = (float **)malloc(WIDTH * sizeof(float *));
    M = (float **)malloc(WIDTH * sizeof(float *));
    N = (float **)malloc(WIDTH * sizeof(float *));
    
    for(int i = 0; i < WIDTH; i++) {
        P[i] = (float *)malloc(WIDTH * sizeof(float));
        M[i] = (float *)malloc(WIDTH * sizeof(float));
        N[i] = (float *)malloc(WIDTH * sizeof(float));
    }

    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < WIDTH; j++) {
            M[i][j] = 1.2;
            N[i][j] = 0.2;
            P[i][j] = 0;
        }
    }

    gettimeofday(&start, NULL);


	#pragma omp parallel for
	for(it = 0; it < ITERATIONS; it++) {
		for(a = 0; a < WIDTH; a += TILE_SIZE) {
			for(b = 0; b < WIDTH; b += TILE_SIZE) {
				for(c = 0; c < WIDTH; c+= TILE_SIZE) {
					for(i = a; i < a + TILE_SIZE; i++)  {
						for(k = c; k < c + TILE_SIZE; k++) {
							for(j = b; j < b + TILE_SIZE; j++) {
								P[i][j] += M[i][k] * N[k][j];
							}
						}
					}
				}
			}
		}
	}

    gettimeofday(&end, NULL);
    totaltime = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / ITERATIONS;
    printf("Mat mul time : %f", totaltime);
    return 0;
}








