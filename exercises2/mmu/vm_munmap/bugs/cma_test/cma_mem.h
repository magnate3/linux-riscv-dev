#ifndef _CMA_MEM_H_
#define _CMA_MEM_H_

#define CMEM_IOCTL_MAGIC 'm'
#define CMEM_GET_PHYS		_IOW(CMEM_IOCTL_MAGIC, 1, unsigned int)
#define CMEM_MAP		_IOW(CMEM_IOCTL_MAGIC, 2, unsigned int)
#define CMEM_GET_SIZE		_IOW(CMEM_IOCTL_MAGIC, 3, unsigned int)
#define CMEM_UNMAP		_IOW(CMEM_IOCTL_MAGIC, 4, unsigned int)

#define CMEM_ALLOCATE		_IOW(CMEM_IOCTL_MAGIC, 5, unsigned int)

#define CMEM_CONNECT		_IOW(CMEM_IOCTL_MAGIC, 6, unsigned int)

#define CMEM_GET_TOTAL_SIZE	_IOW(CMEM_IOCTL_MAGIC, 7, unsigned int)
#define CMEM_CACHE_FLUSH	_IOW(CMEM_IOCTL_MAGIC, 8, unsigned int)

struct mem_block {
	char name[10];
	char is_use_buffer;
	int id;
	unsigned long offset;
	unsigned long len;
	unsigned long phy_base;
	unsigned long mem_base;
	void *kernel_base;
};



#endif
