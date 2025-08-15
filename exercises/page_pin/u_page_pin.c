#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>

struct test_pin_address {
	unsigned long addr;
	unsigned long size;
};

#define TEST_PIN		_IOW('W', 0, struct test_pin_address)
#define TEST_UNPIN		_IOW('W', 1, struct test_pin_address)

int main()
{
	unsigned long size = 2 * 4096;
	unsigned long *p;
	struct test_pin_address addr;
	int fd, ret;

	p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE |
		 MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED) {
		perror("fail to do mmap\n");
		return -1;
	}

	fd = open("/dev/page_pin", O_RDWR);
	if (fd < 0) {
		perror("fail to open\n");
		return -1;
	}

	addr.addr = (unsigned long)p;
	addr.size = size;
	ret = ioctl(fd, TEST_PIN, &addr);
	if (ret < 0) {
		perror("fail to pin\n");
		return -1;
	}

	/* todo: add sleep to check */

	ret = ioctl(fd, TEST_UNPIN, &addr);
	if (ret < 0) {
		perror("fail to unpin\n");
		return -1;
	}

	return 0;
}
