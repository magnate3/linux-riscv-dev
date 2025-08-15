#include<stdio.h>
#include<stdlib.h>
#include "timing.c"
#include "OptimizationSplay.c"

double number[1000000];
void getRandomNumber(){
    FILE *fp = fopen("resource/random_number/biased.txt", "r");
    int index = 0;
    double buff;
    while (!feof (fp)) {
        fscanf(fp, "%lf", &buff);
        number[index] = buff;
        index++;
    }
    fclose(fp);
}

void printFlie(FILE *f, double wc1, double wc2, long count){
    fprintf(f,"%f  %ld\n",(wc2 - wc1)*1000, count);
}

int main(){
    //initqueue();
    getRandomNumber();
    FILE *f = fopen("result/the_hold_model/result.txt","a");
    double wc1 = 0, wc2 = 0, cpuT = 0;
    long count = 0, n = 100;
    long index = 0;
    double current = 0;

    int first = -1;
    unsigned long arr[20250][7];//20250 = 3*(k*k*k/4) as k = 30
    int root = -1;

    // start
    timing(&wc1, &cpuT);

    // begin insert 1000 event
    for(int i=0; i<1000; i++){
        //enqueue(new_node(A,0,0,0));
        enqueue(A, 0, 0, 0, &root, arr);
    }

    while(1){
        //node* del = dequeue();
        //current = del->endTime;
        dequeue(&first, &root, arr); current = first;

        //node* new_n = new_node(A, 0, 0, current + number[index]); index++;
        //enqueue(new_n);
        enqueue(A, 0, 0, current + number[index], &root, arr); index++;

        count++;
        if(index > 1000000) index = 0;
        if(count == n) break;
    }

    // end
    timing(&wc2, &cpuT);

    printf("Time: %f ms with count = %ld\n", (wc2 - wc1)*1000, count);
    printf("================================\n");
    printFlie(f,wc1,wc2,count);
}
