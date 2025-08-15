#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>
#include <stdint.h>

int main(void)
{
	printf("pid :%d\n", getpid());
	int64_t *ptr[10];  
	sleep(20); 	
	getchar();
	for(int i=0;i<10;i++){
		ptr[i] = (int64_t*)malloc(sizeof(int64_t));
	}

	for(int i=0;i<9;i++){
		free(ptr[i]);
	}
	return 0;
}
