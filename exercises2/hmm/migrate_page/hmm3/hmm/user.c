
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
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8
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

uint64_t mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long long virt_pfn;	// virtual page frame number
	int page_size;
	off_t offset;

	/* standard page size */
	page_size = getpagesize();

	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "%s(): cannot open /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}

	virt_pfn = (unsigned long)virtaddr / page_size;
	//printf("Virtual page frame number is %llu\n", virt_pfn);

	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t) - 1) {
		fprintf(stderr, "%s(): seek error in /proc/self/pagemap: %s\n", __func__, strerror(errno));
		close(fd);
		return BAD_PHYS_ADDR;
	}

	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0) {
		fprintf(stderr, "%s(): cannot read /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	} else if (retval != PFN_MASK_SIZE) {
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d\n", 
		        __func__, retval, PFN_MASK_SIZE);
		return BAD_PHYS_ADDR;
	}

	/*
	 * the pfn (page frame number) are bits 0-54 (see
	 * pagemap.txt in linux Documentation)
	 */
	if ((page & 0x7fffffffffffffULL) == 0) {
		fprintf(stderr, "Zero page frame number\n");
		return BAD_PHYS_ADDR;
	}

	physaddr = ((page & 0x7fffffffffffffULL) * page_size) + ((unsigned long)virtaddr % page_size);

	return physaddr;
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
        uint64_t paddr;
        const char * str1 = "hello world";
        struct hmm2 *self;
	char buf[64] = {0};
	self = (struct hmm2 *)malloc(sizeof(struct hmm2));
	init_hmm2(self);
	npages = 1;
        size = npages << self->page_shift;
	buffer = malloc(sizeof(*buffer));
	ASSERT_NE(buffer, NULL);

	buffer->fd = -1;
	buffer->size = size;
	buffer->mirror = malloc(size);
	ASSERT_NE(buffer->mirror, NULL);

	/* Reserve a range of addresses. */
	buffer->ptr = mmap(NULL, size,
			   PROT_NONE,
			   MAP_PRIVATE | MAP_ANONYMOUS,
			   buffer->fd, 0);
	ASSERT_NE(buffer->ptr, MAP_FAILED);
#if 1
        // allow read and write
        ret = mprotect(buffer->ptr, self->page_size, PROT_READ | PROT_WRITE);
	ASSERT_EQ(ret, 0);
	memcpy(buffer->ptr,str1,strlen(str1)+1);
        paddr = mem_virt2phy(buffer->ptr);
        printf("***** before migrate: \n Physical address is %llu\n", (unsigned long long)paddr);
	memcpy(buf, buffer->ptr, sizeof(buf) -1);
	printf("content is %s \n", buf);
#endif
	ret = hmm_dmirror_cmd(self->fd0, HMM_DMIRROR_MIGRATE, buffer, 1);
	ASSERT_EQ(ret, 0);
        paddr = mem_virt2phy(buffer->ptr);
        printf("**** after migrate: \n Physical address is %llu\n", (unsigned long long)paddr);
	memcpy(buf, buffer->ptr, sizeof(buf) -1);
	printf("content is %s \n", buf);
	exit_hmm2(self);
	hmm_buffer_free(buffer);
	free(self);
	printf("run over \n");
	return 0;
}

