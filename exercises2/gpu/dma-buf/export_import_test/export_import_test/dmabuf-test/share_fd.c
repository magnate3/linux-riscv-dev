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

#include "dma_buf_exporter_kmd.h"
int main(int argc, char *argv[])
{
	int fd1,fd2;
	int dmabuf_fd = 0;
        struct dma_exporter_buf_alloc_data alloc_data;
        int size = getpagesize();

	fd1 = open("/dev/exporter", O_RDONLY);
	if (fd1 < 0) {
		printf("open /dev/exporter failed, %s\n", strerror(errno));
		return -1;
	}
        alloc_data.size= size;
        if (ioctl(fd1, DMA_BUF_EXPORTER_ALLOC, &alloc_data)) {
                printf("ion ioctl failed, %s\n", strerror(errno));
                close(fd1);
                return -1;
        }

        dmabuf_fd = alloc_data.fd;
	fd2 = open("/dev/importer", O_RDONLY);
	if (fd2 < 0) {
		printf("open /dev/importer failed, %s\n", strerror(errno));
                goto err1;
	}
	ioctl(fd2, 0, &dmabuf_fd);
	close(fd2);
        if (ioctl(fd1, DMA_BUF_EXPORTER_FREE, &alloc_data)) {
                printf("ion ioctl failed, %s\n", strerror(errno));
                close(fd1);
	        return -1;
        }
        close(fd1);
	return 0;
err1:
        if (ioctl(fd1, DMA_BUF_EXPORTER_FREE, &alloc_data)) {
                printf("ion ioctl failed, %s\n", strerror(errno));
        }
        close(fd1);
        return -1;
}
