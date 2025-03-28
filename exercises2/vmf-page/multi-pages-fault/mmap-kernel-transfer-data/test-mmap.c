#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>

#define NR_PAGES (1 << 2)
int test_write_read( unsigned char *mmap_addr)
{
	int i;
	printf("\nWrite/Read test ...\n");

	/* TODO: write to device mmap'ed address */
	for(i = 0 ; i < NR_PAGES* getpagesize() ; i += getpagesize()){
		mmap_addr[i] = 'h';
		mmap_addr[i+1] = 'l';
		mmap_addr[i+2] = 'l';
		mmap_addr[i+3] = 'o';
		mmap_addr[i+4] = ' ';
		mmap_addr[i+5] = 'w';
		mmap_addr[i+6] = 'o';
		mmap_addr[i+7] = 'r';
		mmap_addr[i+7] = 'l';
		mmap_addr[i+7] = 'd';
		mmap_addr[i+8] = '\0';
	}

	return 0;
}
int main (int argc, char **argv)
{
    int configfd;
    char *address = NULL;
    int total = getpagesize() * NR_PAGES;
    configfd = open ("/dev/mmap-test", O_RDWR);
    if (configfd < 0)
    {
        perror ("Open call failed");
        return -1;
    }

    address = mmap (NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, configfd, 0);
    if (address == MAP_FAILED)
    {
        perror ("mmap operation failed");
        return -1;
    }

    printf ("Initial message: %s\n", address);
    memcpy (address + 11, "*user*", 6);
    printf ("Changed message: %s\n", address);
    test_write_read(address);
    munmap(address, total);
    close (configfd);
  
    return 0;
}
