#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// return random number between 0 and 1
double randomDouble(){
    //srand((double)time(0));
    double scale = (double) rand() / (double) RAND_MAX;
    return scale;
}

// exponential distribution
double exponential(){
    double rnd = randomDouble();
    if(rnd == 0) return 1000000;
    else {
        return -log(rnd);
    }
}

// uniform real number distribution between 0 and x
double uniform(double x){
    return x * randomDouble();
}

// biased real number distribution between x and y
double biased(double x, double y){
    return x + (y-x) * randomDouble();
}

// bimodal distribution
double bimodal(){
    double rnd = randomDouble();
    return 0.95238 * rnd + rnd<0.1? 9.5238 : 0;
}

// triangular distribution between 0 to x
double triangular(double x){
    double rnd = randomDouble();
    return x * sqrt(rnd);
}

int main(){
    FILE *f;
    srand(time(NULL));
    f = fopen("resource/random_number/triangular.txt", "w");
    int i = 0;
    for(i=0; i<1000000; i++){
        int r = (int) triangular(10.0);
        printf("%d\n", r);
        fprintf(f, "%d\n", r);
    }

    f = fopen("resource/random_number/bimodal.txt", "w");
    for(i=0; i<1000000; i++){
        int r = (int) bimodal();
        printf("%d\n", r);
        fprintf(f, "%d\n", r);
    }

    f = fopen("resource/random_number/uniform.txt", "w");
    for(i=0; i<1000000; i++){
        int r = (int) uniform(10.0);
        printf("%d\n", r);
        fprintf(f, "%d\n", r);
    }

    f = fopen("resource/random_number/exponential.txt", "w");
    for(i=0; i<1000000; i++){
        int r = (int) exponential();
        printf("%d\n", r);
        fprintf(f, "%d\n", r);
    }

    f = fopen("resource/random_number/biased.txt", "w");
    for(i=0; i<1000000; i++){
        int r = (int) biased(0, 10.0);
        printf("%d\n", r);
        fprintf(f, "%d\n", r);
    }

    f = fopen("resource/random_variate/variate.txt", "w");
    for(i=0; i<1000000; i++){
        double r = randomDouble();
        printf("%lf\n", r);
        fprintf(f, "%lf\n", r);
    }
    fclose(f);
}
