#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#define ONE p = (char **)*p;
#define FIVE    ONE ONE ONE ONE ONE
#define TEN FIVE FIVE
#define FIFTY   TEN TEN TEN TEN TEN
#define HUNDRED FIFTY FIFTY
static void usage()
{
    printf("Usage: ./mem-lat -b xxx -n xxx -s xxx\n");
    printf("   -b buffer size in KB\n");
    printf("   -n number of read\n\n");
    printf("   -s stride skipped before the next access\n\n");
    printf("Please don't use non-decimal based number\n");
}
int main(int argc, char* argv[])
{
  unsigned long i, j, size, tmp;
    unsigned long memsize = 0x800000; /* 1/4 LLC size of skylake, 1/5 of broadwell */
    unsigned long count = 1048576; /* memsize / 64 * 8 */
    unsigned int stride = 64; /* skipped amount of memory before the next access */
    unsigned long sec, usec;
    struct timeval tv1, tv2;
    struct timezone tz;
    unsigned int *indices;
    while (argc-- > 0) {
        if ((*argv)[0] == '-') {  /* look at first char of next */
            switch ((*argv)[1]) {   /* look at second */
                case 'b':
                    argv++;
                    argc--;
                    memsize = atoi(*argv) * 1024;
                    break;
                case 'n':
                    argv++;
                    argc--;
                    count = atoi(*argv);
                    break;
                case 's':
                    argv++;
                    argc--;
                    stride = atoi(*argv);
                    break;
                default:
                    usage();
                    exit(1);
                    break;
            }
        }
        argv++;
    }
  char* mem = mmap(NULL, memsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    // trick3: init pointer chasing, per stride=8 byte
    size = memsize / stride;
    indices = malloc(size * sizeof(int));
    for (i = 0; i < size; i++)
        indices[i] = i;
    // trick 2: fill mem with pointer references
    for (i = 0; i < size - 1; i++)
        *(char **)&mem[indices[i]*stride]= (char*)&mem[indices[i+1]*stride];
    *(char **)&mem[indices[size-1]*stride]= (char*)&mem[indices[0]*stride];
    register char **p = (char **) mem;
    //char **p = (char **) mem;
    tmp = count / 100;
    gettimeofday (&tv1, &tz);
    for (i = 0; i < tmp; ++i) {
        HUNDRED;  //trick 1
    }
    gettimeofday (&tv2, &tz);
    char **touch = p;
    if (tv2.tv_usec < tv1.tv_usec) {
        usec = 1000000 + tv2.tv_usec - tv1.tv_usec;
        sec = tv2.tv_sec - tv1.tv_sec - 1;
    } else {
        usec = tv2.tv_usec - tv1.tv_usec;
        sec = tv2.tv_sec - tv1.tv_sec;
    }
    printf("Buffer size: %ld KB, stride %u, time %ld.%06ld s, latency %.2f ns\n",
            memsize/1024, stride, sec, usec, (sec * 1000000  + usec) * 1000.0 / (tmp *100));
    munmap(mem, memsize);
    free(indices);
}
