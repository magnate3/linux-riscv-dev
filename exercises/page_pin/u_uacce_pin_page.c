#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>

struct uacce_pin_address {
	unsigned long addr;
	unsigned long size;
};

#define UACCE_CMD_PIN		_IOW('W', 2, struct uacce_pin_address)
#define UACCE_CMD_UNPIN		_IOW('W', 3, struct uacce_pin_address)

int write_done = 0;
char *p;

static void *write_data(void *arg)
{
	cpu_set_t mask;
	/* will make 0-3 cpu in node0 and 4-7 in node1 in qemu machine */
	int cpu_id = 4;
	int i;

	printf("write data is thread %ld\n", syscall(__NR_gettid));
	CPU_ZERO(&mask);
	CPU_SET(cpu_id, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
		perror("fail to set pthread affinity\n");
		return NULL;
	}

	while (!write_done);
	while (1) {
		for (i = 0; i < 100 * 4096; i++)
			*(p + i) = 9;
	}

	return NULL;
}

int main()
{
	unsigned long size = 100 * 4096;
	struct uacce_pin_address addr;
	struct rusage u1, u2;
	int i, fd, ret = 0;
	pthread_t id;

 	pthread_create(&id, NULL, write_data, NULL);

	p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE |
		 MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED) {
		perror("fail to do mmap\n");
		return -1;
	}

	for (i = 0; i < 100; i++)
		*(p + i * 4096) = 7;

	fd = open("/dev/uacce_pin_page", O_RDWR);
	if (fd < 0) {
		perror("fail to open\n");
		return -1;
	}

	addr.addr = (unsigned long)p;
	addr.size = size;
	printf("u pin: %lx %lx\n", p, size);

	getrusage(RUSAGE_SELF, &u1);
	printf("min page fault before pin: %ld\n", u1.ru_minflt);

	/* comments out this line to see page migiration */
//	ret = ioctl(fd, UACCE_CMD_PIN, &addr);
	if (ret < 0) {
		perror("fail to pin\n");
		return -1;
	}

	getrusage(RUSAGE_SELF, &u2);
	printf("min page fault after pin: %ld\n", u2.ru_minflt);

	sleep(10);
	write_done = 1;
	while (1);

//	ret = ioctl(fd, UACCE_CMD_UNPIN, &addr);
	if (ret < 0) {
		perror("fail to unpin\n");
		return -1;
	}

	return 0;
}

/* gcc u_uacce_pin_page.c -lpthread --static */
