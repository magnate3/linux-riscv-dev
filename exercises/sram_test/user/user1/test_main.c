#include<stdio.h>
#include<malloc.h>
#include"dlmalloc.h"
#define   MALLOC_SIZE  2048
#define   DLMALLOC_SIZE  512
int main()
{
        char * base = (char*)malloc(MALLOC_SIZE);
        char *p;
        printf("base addr %p \n", base);
        p=create_mspace_with_base((void *)base,MALLOC_SIZE,0);
        printf("dlmalloc addr %p \n", p);
        destroy_mspace(p);
        free(base);
        return 0;
}
