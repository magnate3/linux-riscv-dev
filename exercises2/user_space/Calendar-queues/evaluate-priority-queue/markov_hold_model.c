#include<stdio.h>
#include<stdlib.h>
#include "timing.c"
#include "VoidSplay.c"
#include <locale.h>
#include <math.h>

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

double variate[1000000];
void getRandomVariate(){
    FILE *fp = fopen("resource/random_variate/variate.txt", "r");
    int index = 0;
    double buff;
    while (!feof (fp)) {
        fscanf(fp, "%lf", &buff);
        variate[index] = buff;
        index++;
    }
    fclose(fp);
}

void printFlie(FILE *f, double wc1, double wc2, long count, ProcStatm proc_statm, long page_size){
    fprintf(f,"%f  %ld\n",(wc2 - wc1)*1000, count);

    double time = wc2 - wc1;
    ProcStat_init(&proc_statm);
    double total_vm = ((double)proc_statm.size * page_size) / (1024 * 1024);
    fprintf(f,"/proc/self/statm size resident %f MiB, page_size %ld\n",
                total_vm, page_size
            );
    double minutes = time / 60;
    if(time < 1)
	time = time * 1000;
    double _badness = total_vm / (int_sqrt((long)time) * //pow(minutes, 1.0/4)
                        sqrt(sqrt(minutes))
                        );
    fprintf(f,"Badness bd = %f\n", _badness);
}

/*MAIN*/
int main(int argc, char** argv){
    /*init priority queue*/
    // khoi tao calendar queue
    //initqueue();

    // khoi tao array splay
    //int root = -1, first = -1;
    //unsigned long arr[MARCO][7];//20250 = 3*(k*k*k/4) as k = 30

    // khoi tao void splay
    splay_tree *t = new_splay_tree();

    getRandomNumber();
    getRandomVariate();
    FILE *f = fopen("result/markov_hold_model/result.txt", "a");
    /*end*/

    /*init variable*/
    double wc1 = 0, wc2 = 0, cpuT = 0;
    int defaultBias = 1;
    ProcStatm proc_statm;
    long page_size = sysconf(_SC_PAGESIZE);

    if(argc >= 2)
    {
        if(argc >= 3)
            defaultBias = atoi(argv[2]);
    }

    long count = 0, n = 10000000;
    long index = 0;
    int current = 0;
    int i = 0;
    setlocale(LC_NUMERIC, "");

    unsigned long mem = mem_avail();
    printf("Free memory Available = %'ld\n", mem / (1024*1024));
    printf("Start......\n");

    /*variable for state markov hold model*/
    int state = 0; // delete
    int done = 0; // run program
    int variateIndex = 0;
    double anpha = 0.5;
    double beta = 0.5;
    /*end*/

    /*START*/
    timing(&wc1, &cpuT);

    // begin insert 1000 event
    for(i=0; i<10000; i++){
        enqueue(t, new_node(A, i, 0L, 0));

        //enqueue(A, i, 0, 0, &root, arr);

        //enqueue(new_node(A,0,0,0));
    }

    node * del = dequeue(t); current = del->endTime; //printf("%d \n",  current);

    //unsigned long del;
    //dequeue(&first, &root, arr); del = arr[first][3]; current = del;

    //node* del = dequeue(); current = del->endTime;
    while(!done){
        if(state == 0){
            if(/*qsize == 0*//*first == -1*/t == NULL){ // queue is empty
                printf("error!");
                return 0;
            }

            double randomU = variate[variateIndex]; // get U(0,1) random variate u
            variateIndex++; if(variateIndex == 1000000) variateIndex = 0;

            if(randomU < anpha){
                del = dequeue(t); current = del->endTime; //printf("%d \n",  current);

                //dequeue(&first, &root, arr); del = arr[first][3]; current = del;

                //del = dequeue(); current = del->endTime;
                state = 0;
                count++;
            } else {
                int i = del->idElementInGroup;
                enqueue(t, new_node(A, i, 0L, current + number[index]));

                //int i = arr[first][1];
                //enqueue(A, i, 0, current + number[index], &root, arr);

                //node* new_n = new_node(A, 0, 0, current + number[index]);
                //enqueue(new_n);
                index++; if(index == 1000000) index = 0;
                state = 1;
                count++;
            }
        } else {
            double randomU = variate[variateIndex]; // get U(0,1) random variate u
            variateIndex++; if(variateIndex == 1000000) variateIndex = 0;

            if(randomU < 1 - beta){
                del = dequeue(t); current = del->endTime; //printf("%d \n",  current);

                //dequeue(&first, &root, arr); del = arr[first][3]; current = del;

                //del = dequeue(); current = del->endTime;
                state = 0;
                count++;
            } else {
                int i = del->idElementInGroup;
                enqueue(t, new_node(A, i, 0L, current + number[index]));

                //int i = arr[first][1];
                //enqueue(A, i, 0, current + number[index], &root, arr);

                //node* new_n = new_node(A, 0, 0, current + number[index]);
                //enqueue(new_n);
                index++; if(index == 1000000) index = 0;
                state = 1;
                count++;
            }
        }

        if(count == n){
            done = 1;
        }
    }


    timing(&wc2, &cpuT);
    /*END*/

    printf("Time: %'f ms with count = %'ld\n", (wc2 - wc1)*1000, count);
    printf("================================\n");
    badness(wc2 - wc1, page_size, proc_statm);
    printFlie(f,wc1,wc2,count, proc_statm,page_size);

    return 0;
}

