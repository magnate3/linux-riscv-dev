#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define DEV_NAME "/dev/simple"

void create_data(void) 
{ 
    int fd, i;

    fd = open(DEV_NAME, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    for (i = 0; i < 10; i++)
        write(fd, &i, sizeof(int));
    close(fd);
}

void display_data(void)
{
    int fd = 0, data = 0, i;
    
    fd = open(DEV_NAME, O_RDONLY);
    for (i = 0; i < 10; i++)
        if (read(fd, &data, sizeof(int)) == 4)
            printf("%4d", data);
    puts(""); 
    close(fd);
}

void mmap_data(void)
{
    int *mapped = NULL;
    int fd;
    
    fd = open(DEV_NAME, O_RDWR);
    mapped = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == NULL)
        return;
    
    mapped[7] += 200;
    msync(mapped, 4096, MS_ASYNC);
    munmap(mapped, 4096);
    close(fd);
}

int main(void)
{
    create_data();
    display_data();
    mmap_data();
    display_data();
    return 0;
}

