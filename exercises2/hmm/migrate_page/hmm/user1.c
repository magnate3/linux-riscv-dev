
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <malloc.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "test_hmm_user_uapi.h"
struct hmm_buffer {
	void		*ptr;
	void		*mirror;
	unsigned long	size;
	int		fd;
	uint64_t	cpages;
	uint64_t	faults;
};

#define TWOMEG		(1 << 21)
#define HMM_BUFFER_SIZE (1024 << 12)
#define HMM_PATH_MAX    64
#define NTIMES		10

#define ALIGN(x, a) (((x) + (a - 1)) & (~((a) - 1)))
struct hmm1
{
	int		fd;
	unsigned int	page_size;
	unsigned int	page_shift;
};

struct hmm2
{
	int		fd0;
	int		fd1;
	unsigned int	page_size;
	unsigned int	page_shift;
};
static int hmm_open(int unit)
{
	char pathname[HMM_PATH_MAX];
	int fd;

	snprintf(pathname, sizeof(pathname), "/dev/hmm_dmirror%d", unit);
	fd = open(pathname, O_RDWR, 0);
	if (fd < 0)
		fprintf(stderr, "could not open hmm dmirror driver (%s)\n",
			pathname);
	return fd;
}
#define ASSERT_NE(arg1,arg2)  do {\
     if((arg2) == (arg1))\
     {\
	 printf("illeagal equal \n");\
	 exit(0);\
     }\
}while(0)
#define ASSERT_EQ(arg1,arg2)  do {\
     if((arg2) != (arg1))\
     {\
	 printf("not equal \n");\
	 exit(0);\
     }\
}while(0)
#define ASSERT_GE(arg1,arg2)  do {\
     if((arg2) >= (arg1))\
     {\
	 printf("not great \n");\
	 exit(0);\
     }\
}while(0)
void init_hmm1(struct hmm1 * self)
{
	//self->page_size = sysconf(_SC_PAGE_SIZE);
	self->page_size = getpagesize();
	self->page_shift = ffs(self->page_size) - 1;

	self->fd = hmm_open(0);
	ASSERT_GE(self->fd, 0);
}

void init_hmm2(struct hmm2 * self)
{
	//self->page_size = sysconf(_SC_PAGE_SIZE);
	self->page_size = getpagesize();
	self->page_shift = ffs(self->page_size) - 1;

	self->fd0 = hmm_open(0);
	ASSERT_GE(self->fd0, 0);
	self->fd1 = hmm_open(1);
	ASSERT_GE(self->fd1, 0);
}


void exit_hmm1(struct hmm1 * self)
{
	int ret = close(self->fd);

	ASSERT_EQ(ret, 0);
	self->fd = -1;
}

void exit_hmm2(struct hmm2 * self)
{
	int ret = close(self->fd0);

	ASSERT_EQ(ret, 0);
	self->fd0 = -1;

	ret = close(self->fd1);
	ASSERT_EQ(ret, 0);
	self->fd1 = -1;
}

static int hmm_dmirror_cmd(int fd,
			   unsigned long request,
			   struct hmm_buffer *buffer,
			   unsigned long npages)
{
	struct hmm_dmirror_cmd cmd;
	int ret;

	/* Simulate a device reading system memory. */
	cmd.addr = (__u64)buffer->ptr;
	cmd.ptr = (__u64)buffer->mirror;
	cmd.npages = npages;

	for (;;) {
		ret = ioctl(fd, request, &cmd);
		if (ret == 0)
			break;
		if (errno == EINTR)
			continue;
		return -errno;
	}
	buffer->cpages = cmd.cpages;
	buffer->faults = cmd.faults;

	return 0;
}

static void hmm_buffer_free(struct hmm_buffer *buffer)
{
	if (buffer == NULL)
		return;

	if (buffer->ptr)
		munmap(buffer->ptr, buffer->size);
	free(buffer->mirror);
	free(buffer);
}

/*
 * Create a temporary file that will be deleted on close.
 */
static int hmm_create_file(unsigned long size)
{
	char path[HMM_PATH_MAX];
	int fd;

	strcpy(path, "/tmp");
	fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (fd >= 0) {
		int r;

		do {
			r = ftruncate(fd, size);
		} while (r == -1 && errno == EINTR);
		if (!r)
			return fd;
		close(fd);
	}
	return -1;
}

/*
 * Return a random unsigned number.
 */
static unsigned int hmm_random(void)
{
	static int fd = -1;
	unsigned int r;

	if (fd < 0) {
		fd = open("/dev/urandom", O_RDONLY);
		if (fd < 0) {
			fprintf(stderr, "%s:%d failed to open /dev/urandom\n",
					__FILE__, __LINE__);
			return ~0U;
		}
	}
	read(fd, &r, sizeof(r));
	return r;
}

static void hmm_nanosleep(unsigned int n)
{
	struct timespec t;

	t.tv_sec = 0;
	t.tv_nsec = n;
	nanosleep(&t, NULL);
}

int main()
{
	struct hmm_buffer *buffer;
	unsigned long npages;
	unsigned long size;
	int *ptr;
	unsigned char *p;
	unsigned char *m;
	int ret;
	int val;

        struct hmm2 *self;
	self = (struct hmm2 *)malloc(sizeof(struct hmm2));
	init_hmm2(self);
	npages = 7;
        size = npages << self->page_shift;
        //size =getpagesize()*npages;
	buffer = malloc(sizeof(*buffer));
	ASSERT_NE(buffer, NULL);

	buffer->fd = -1;
	buffer->size = size;
	buffer->mirror = malloc(size);
	//buffer->mirror = malloc(npages);
	ASSERT_NE(buffer->mirror, NULL);

	/* Reserve a range of addresses. */
	buffer->ptr = mmap(NULL, size,
			   PROT_NONE,
			   MAP_PRIVATE | MAP_ANONYMOUS,
			   buffer->fd, 0);
	ASSERT_NE(buffer->ptr, MAP_FAILED);
	p = buffer->ptr;

	/* Punch a hole after the first page address. */
	ret = munmap(buffer->ptr + self->page_size, self->page_size);
	ASSERT_EQ(ret, 0);

	/* Page 2 will be read-only zero page. */
	ret = mprotect(buffer->ptr + 2 * self->page_size, self->page_size,
				PROT_READ);
	ASSERT_EQ(ret, 0);
	ptr = (int *)(buffer->ptr + 2 * self->page_size);
	val = *ptr + 3;
	ASSERT_EQ(val, 3);

	/* Page 3 will be read-only. */
	ret = mprotect(buffer->ptr + 3 * self->page_size, self->page_size,
				PROT_READ | PROT_WRITE);
	ASSERT_EQ(ret, 0);
	ptr = (int *)(buffer->ptr + 3 * self->page_size);
	*ptr = val;
	ret = mprotect(buffer->ptr + 3 * self->page_size, self->page_size,
				PROT_READ);
	ASSERT_EQ(ret, 0);

	/* Page 4-6 will be read-write. */
	ret = mprotect(buffer->ptr + 4 * self->page_size, 3 * self->page_size,
				PROT_READ | PROT_WRITE);
	ASSERT_EQ(ret, 0);
	ptr = (int *)(buffer->ptr + 4 * self->page_size);
	*ptr = val;

	/* Page 5 will be migrated to device 0. */
	buffer->ptr = p + 5 * self->page_size;
	ret = hmm_dmirror_cmd(self->fd0, HMM_DMIRROR_MIGRATE, buffer, 1);
	ASSERT_EQ(ret, 0);
	ASSERT_EQ(buffer->cpages, 1);

	/* Page 6 will be migrated to device 1. */
	buffer->ptr = p + 6 * self->page_size;
	ret = hmm_dmirror_cmd(self->fd1, HMM_DMIRROR_MIGRATE, buffer, 1);
	ASSERT_EQ(ret, 0);
	ASSERT_EQ(buffer->cpages, 1);

	/* Simulate a device snapshotting CPU pagetables. */
	buffer->ptr = p;
	ret = hmm_dmirror_cmd(self->fd0, HMM_DMIRROR_SNAPSHOT, buffer, npages);
	ASSERT_EQ(ret, 0);
	ASSERT_EQ(buffer->cpages, npages);

	/* Check what the device saw. */
	m = buffer->mirror;
#if 1
	ASSERT_EQ(m[0], HMM_DMIRROR_PROT_ERROR);
	ASSERT_EQ(m[1], HMM_DMIRROR_PROT_ERROR);
	ASSERT_EQ(m[2], HMM_DMIRROR_PROT_ZERO | HMM_DMIRROR_PROT_READ);
	ASSERT_EQ(m[3], HMM_DMIRROR_PROT_READ);
	ASSERT_EQ(m[4], HMM_DMIRROR_PROT_WRITE);
	ASSERT_EQ(m[5], HMM_DMIRROR_PROT_DEV_PRIVATE_LOCAL |
			HMM_DMIRROR_PROT_WRITE);
	ASSERT_EQ(m[6], HMM_DMIRROR_PROT_NONE);
#endif
	exit_hmm2(self);
	hmm_buffer_free(buffer);
	free(self);
	printf("run over \n");
	return 0;
}

