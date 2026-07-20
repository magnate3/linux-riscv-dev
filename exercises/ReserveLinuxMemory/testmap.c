#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>

#define RAW_DATA_SIZE 31457280

int main(int argc, char **argv) {
    int configfd;
    char * address = NULL;
    unsigned long chkSum;
    FILE *fp = fopen("results.log", "w+");

    configfd = open("/sys/kernel/debug/mmap_example", O_RDWR);
    if (configfd < 0) {
        perror("Open call failed");
        return -1;
    }

    address = (unsigned char*) mmap(NULL, RAW_DATA_SIZE, PROT_READ|PROT_WRITE,
            MAP_PRIVATE, configfd, 0);
    if (address == MAP_FAILED) {
        perror("mmap operation failed");
        return -1;
    }

    // for (int i=0; i<RAW_DATA_SIZE;i++)
    //{
    //	address[i]='A';
    // }
    
    
   fputs(address, fp);
    fclose(fp);

    //    printf("%s",address);

    
    close(configfd);
    return 0;
}
