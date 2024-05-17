#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
//#define _BSD_SOURCE             /* Get mincore() declaration and MAP_ANONYMOUS
//                                   definition from <sys/mman.h> */
#include <sys/mman.h>
static void displayMincore(char *addr, size_t length)
{
    unsigned char *vec;
    long pageSize, numPages, j;

#ifndef _SC_PAGESIZE
    pageSize = getpagesize();   /* Some systems don't have _SC_PAGESIZE */
#else
    pageSize = sysconf(_SC_PAGESIZE);
#endif

    numPages = (length + pageSize - 1) / pageSize;
    vec = (unsigned char *)malloc(numPages);
    if (vec == NULL){
        perror("malloc");
        exit(EXIT_FAILURE);
    }


    if (mincore(addr, length, vec) == -1){
        perror("mincore");
        exit(EXIT_FAILURE);
    }


    for (j = 0; j < numPages; j++) {
        if (j % 64 == 0)
            printf("%s%10p: ", (j == 0) ? "" : "\n", addr + (j * pageSize));
        printf("%c", (vec[j] & 1) ? '*' : '.');
    }
    printf("\n");

    free(vec);
}
int main(int argc, char *argv[])
{
    char *addr;
    size_t len, lockLen;
    long pageSize, stepSize, j;

    if (argc != 4 || strcmp(argv[1], "--help") == 0){
        printf("%s num-pages lock-page-step lock-page-len\n", argv[0]);
        exit(EXIT_FAILURE);
    }


#ifndef _SC_PAGESIZE
    pageSize = getpagesize();
    if (pageSize == -1){
        printf("getpagesize");
        exit(EXIT_FAILURE);
    }

#else
    pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize == -1){
        printf("sysconf(_SC_PAGESIZE)");
        exit(EXIT_FAILURE);
    }
#endif

    len =      atoi(argv[1]) * pageSize;  //"num-pages"
    stepSize = atoi(argv[2]) * pageSize;  //"lock-page-step"
    lockLen =  atoi(argv[3]) * pageSize;  //"lock-page-len"

    addr = (char *)mmap(NULL, len, PROT_READ, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED){
        perror("mmap");
        exit(EXIT_FAILURE);
    }


    printf("Allocated %ld (%#lx) bytes starting at %p\n",
           (long) len, (unsigned long) len, addr);
    /*FIXME: above: should use %zu here, and remove (long) cast */

    printf("Before mlock:\n");
    displayMincore(addr, len);

    /* Lock pages specified by command-line arguments into memory */

    for (j = 0; j + lockLen <= len; j += stepSize)
        if (mlock(addr + j, lockLen) == -1){
            perror("mlock");
            exit(EXIT_FAILURE);
        }


    printf("After mlock:\n");
    displayMincore(addr, len);

    exit(EXIT_SUCCESS);
}
