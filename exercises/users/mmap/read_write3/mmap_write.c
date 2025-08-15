#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/ipc.h>
#include<sys/msg.h>
#include<sys/types.h>
#include<unistd.h>
#include<errno.h>
#include<fcntl.h>
#include<sys/stat.h>
#include<sys/mman.h>

#define ERR_EXIT(m) \
    do { \
        perror(m); \
        exit(EXIT_FAILURE); \
    } while(0)

typedef struct stu
{
    char name[4];
    int age;
} STU;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int fd;
    void* goal_addr = (void*)0x10000;
    fd = open(argv[1], O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd == -1)
        ERR_EXIT("open");

    lseek(fd, sizeof(STU) * 5 - 1, SEEK_SET);
    write(fd, "", 1);

    STU *p;
    p = (STU *)mmap(goal_addr, sizeof(STU) * 5, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, getpagesize());

    if (p == -1)
        ERR_EXIT("mmap");
     printf("mapped addres %p and goal_addr %p\n", p, goal_addr);
    char ch = 'a';
    int i;
    for (i = 0; i < 5; i++)
    {
        memcpy((p + i)->name, &ch, 1);
        (p + i)->age = 20 + i;
        ch++;
        printf("age %d", 20 +i);
    }

    printf("initialize over\n");

    munmap(p, sizeof(STU) * 5);
    printf("exit...\n");
    return 0;
}
