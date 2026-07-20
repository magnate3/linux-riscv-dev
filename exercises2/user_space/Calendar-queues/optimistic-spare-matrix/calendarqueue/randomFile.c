#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int my_random(int minN, int maxN){
    return minN + rand() % (maxN + 1 - minN);
}

void loadArray(int a[1000]){
   int num;
   FILE *fptr;

   if ((fptr = fopen("random1000.txt","r")) == NULL){
       printf("Error! opening file");
       return;
   }
   int i = 0;
   while(!feof(fptr)){
    fscanf(fptr,"%d", &num);
    a[i] = num;
    i++;
   }

   fclose(fptr);
}

int main(){
    srand((int)time(0));
    FILE *fptr;
    fptr = fopen("random1000.txt","w");
    int r;
    for(int i = 0; i < 1000; ++i){
        r = my_random(1,100);
        if(r <= 21) r = 0;
        fprintf(fptr,"%d\n", r);
        //printf("%d ",r);
        //if(i % 10 == 0 && i != 0) printf("\n");
    }
    fclose(fptr);
    int a[1000];
    loadArray(a);
    int i = 0;
    for(i = 0; i < 15; i++)
        printf("%d \n", a[i]);
    return 0;
}
