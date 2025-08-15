#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include "dma_buf_exporter_kmd.h"

#define PAGE_SIZE 4096

int main(int argc, char *argv[])
{
	int fd;
	struct dma_exporter_buf_alloc_data alloc_data;

	fd = open("/dev/dma_buf_exporter", O_RDWR);
	if (fd < 0) {
		printf("open /dev/ion failed, %s\n", strerror(errno));
		return -1;
	}

	alloc_data.size= 3 * PAGE_SIZE;
	if (ioctl(fd, DMA_BUF_EXPORTER_ALLOC, &alloc_data)) {
		printf("ion ioctl failed, %s\n", strerror(errno));
		close(fd);
		return -1;
	}

	printf("ion alloc success: size = %llu, dmabuf_fd = %u\n",
		alloc_data.size, alloc_data.fd);

	if (ioctl(fd, DMA_BUF_EXPORTER_FREE, &alloc_data)) {
		printf("ion ioctl failed, %s\n", strerror(errno));
		close(fd);
		return -1;
	}
	close(fd);

	return 0;
}
