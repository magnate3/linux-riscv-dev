#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <linux/nvme_ioctl.h>
#include "linux/nvme.h"
#define NVME_DEVICE_PATH "/dev/nvme1n1"  // NVMe设备路径
int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Need at least one argument to write to the file\n");
        exit(EXIT_FAILURE);
    }
    
    const char *text = argv[1];
    printf("Will write text '%s'\n", text);
        
    /* Open a file for writing.
     *  - Creating the file if it doesn't exist.
     *  - Truncating it to 0 size if it already exists. (not really needed)
     *
     * Note: "O_WRONLY" mode is not sufficient when mmaping.
     */
    
    const char *filepath = "/dev/p2pmem0";

    int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    int fd2 = 0;
    
    if (fd == -1)
    {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    fd2 = open(NVME_DEVICE_PATH, O_RDWR);  // 打开NVMe设备文件
    if (fd2 == -1) {
        perror("Failed to open NVMe device");
	close(fd);
        return 1;
    }

    // Stretch the file size to the size of the (mmapped) array of char

    size_t textsize = strlen(text) + 1; // + \0 null character
    
    int count = 0;
// can not do lseek
#if 0
    if (lseek(fd, textsize-1, SEEK_SET) == -1)
    {
        close(fd);
        perror("Error calling lseek() to 'stretch' the file");
        exit(EXIT_FAILURE);
    }
   
    /* Something needs to be written at the end of the file to
     * have the file actually have the new size.
     * Just writing an empty string at the current file position will do.
     *
     * Note:
     *  - The current position in the file is at the end of the stretched 
     *    file due to the call to lseek().
     *  - An empty string is actually a single '\0' character, so a zero-byte
     *    will be written at the last byte of the file.
     */
    
    if (write(fd, "", 1) == -1)
    {
        close(fd);
        perror("Error writing last byte of the file");
        exit(EXIT_FAILURE);
    }
#endif   

    struct nvme_user_io io;
    memset(&io, 0, sizeof(struct nvme_user_io));
    // Now the file is ready to be mmapped.
    char *map = mmap(0, textsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED)
    {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }
    
    for (size_t i = 0; i < textsize; i++)
    {
        printf("Writing character %c at %zu\n", text[i], i);
        map[i] = text[i];
    }

    io.opcode = nvme_cmd_write;  // 写操作
    io.addr = (unsigned long)map;  // 分配缓冲区，假设大小为4KB
    io.slba = 0;  // 起始逻辑块地址（LBA）
    io.nblocks = 1;  // 操作的块数，这里只写入一个块
#if 0
    // 发送命令到NVMe设备
    if (ioctl(fd2, NVME_IOCTL_SUBMIT_IO, &io) == -1) {
        perror("Failed to submit IO request");
        close(fd);
        close(fd2);
	munmap(map,textsize);
        return 1;
    }
#else
		count = pwrite(fd2, map, textsize, 0);
		if (count == -1) {
			goto error2;
		}
#endif
    printf("Write operation completed successfully\n");
// can not do msyc
#if 0    
    // Write it now to disk
    if (msync(map, textsize, MS_SYNC) == -1)
    {
        perror("Could not sync the file to disk");
    }
#endif 
    // Don't forget to free the mmapped memory
    if (munmap(map, textsize) == -1)
    {
        close(fd);
        perror("Error un-mmapping the file");
        exit(EXIT_FAILURE);
    }

    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
    close(fd2);
    
    return 0;
error2:
    close(fd);
    close(fd2);
    munmap(map,textsize);
    return 0;
}
