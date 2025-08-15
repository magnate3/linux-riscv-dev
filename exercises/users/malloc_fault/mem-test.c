#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#define BUF_SIZE 838860800
#define STEP 2097152 //2MBint main(){
int main(int argc, char* argv[])
{
    int i = 0, count = 0;
    char *p = (char*)malloc(BUF_SIZE);
    sleep(15);
    for(i = 0; i < BUF_SIZE; i++){
	    memset(p + i, 0, 4096);
	    i += STEP;
	    count++;
    }
    printf("done, count=%d\n", count);
    //sleep(60);
    getchar();
    free(p);
    return 0;
}
