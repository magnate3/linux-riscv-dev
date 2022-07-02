#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#define MAP_SIZE 1024*1024
int main(void){
        char  *p;
	pid_t pid;
        int i;

	p = mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_ANON | MAP_PRIVATE |  MAP_HUGETLB, -1, 0);
	if(p == MAP_FAILED){	
		perror("mmap error");
		exit(1);
	}
        for (i =0 ;  i <  MAP_SIZE; ++i)
        {
             p[i]= 99;
        } 
#if 0
	if(madvise(p, MAP_SIZE ,MADV_MERGEABLE))
	{
	   printf("faile to enable ksm \n");
	}
	{
	   printf(" enable ksm \n");
	}
#endif
	while(1);
	munmap(p, MAP_SIZE);				//ÊÍ·ÅÓ³ÉäÇø
	return 0;
}
