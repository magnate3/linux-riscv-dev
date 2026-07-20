/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "dma_buf_exporter_kmd.h"

int main(int argc, char *argv[])
{
	int fd;
	int dmabuf_fd = 0;
        struct dma_exporter_buf_alloc_data alloc_data;
        int size = getpagesize();
	fd = open("/dev/exporter", O_RDONLY);
	if (fd < 0) {
		printf("open /dev/exporter failed, %s\n", strerror(errno));
		return -1;
	}

        alloc_data.size= size;
        if (ioctl(fd, DMA_BUF_EXPORTER_ALLOC, &alloc_data)) {
                printf("ion ioctl failed, %s\n", strerror(errno));
                close(fd);
                return -1;
        }

        printf("ion alloc success: size = %llu, dmabuf_fd = %u\n",
                alloc_data.size, alloc_data.fd);
          
        dmabuf_fd = alloc_data.fd;
        printf("dma buf fd %d \n", dmabuf_fd);
        if (dmabuf_fd < 0){
            goto err1;
        }
 
	char *str = mmap(NULL, size, PROT_READ, MAP_SHARED, dmabuf_fd, 0);
	if (str == MAP_FAILED) {
		printf("mmap dmabuf failed: %s\n", strerror(errno));
                goto err1;
	}
	printf("read from dmabuf mmap: %s\n", str);
        munmap(str,size);
        if (ioctl(fd, DMA_BUF_EXPORTER_FREE, &alloc_data)) {
                printf("ion ioctl failed, %s\n", strerror(errno));
                close(fd);
                return -1;
        }

	close(dmabuf_fd);
	close(fd);
	return 0;
err1:
	close(fd);
	return -1;
}
