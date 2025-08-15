#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>


void die(const char *fmt, ...)
{
	va_list ap;
	fprintf(stderr, "ERROR: ");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	exit(1);
}

int main(void)
{
	uint32_t *data;
	int fd, rc;
	size_t size = 1024 * 1024;

	fd = open("/dev/mr", O_RDWR);
	if (fd < 1) die("open(/dev/mr): %s\n", strerror(errno));

	data = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
			0);
	if (fd < 1)
		die("mmap(/dev/mr): %s\n", strerror(errno));

	printf("map[0] = 0x%08x\n", data[0]);

	data[0] = 1;

	printf("map[0] = 0x%08x\n", data[0]);

	//rc = msync(data, sizeof(data[0]), MS_SYNC);
	//if (rc)
	//	die("msync(/dev/mr): %s\n", strerror(errno));

	printf("Doing ioctl\n");
	ioctl(fd, 55, NULL);

	sleep(1);
	printf("map[0] = 0x%08x\n", data[0]);
	sleep(1);
	printf("Modifying data[0]\n");
	data[0] = 2;

	sleep(1);


	close(fd);
	munmap(data, size);
	return 0;
}
