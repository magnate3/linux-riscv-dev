#define _XOPEN_SOURCE 700
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> /* uintmax_t */
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h> /* sysconf */
#include <time.h>

#include "common.h"
#include "zlog_ring_buf.h"

// https://github.com/cirosantilli/linux-kernel-module-cheat/tree/master/kernel_module
// http://egloos.zum.com/furmuwon/v/11206485
// http://wenx05124561.blog.163.com/blog/static/1240008052013101715644618/
// http://chocokeki.tistory.com/672

int main(int argc, char **argv)
{
	int fd = -1, ret = -1;
	long page_size;
	char *addr = NULL;
	zlog_read_t zread;
	uintptr_t paddr;
	uint32_t len, mapped_size;
    uint32_t i;
	zlog_ring_t *zring;
	char *proc_name = "/proc/zlog";

	if (argc >= 2) {
		proc_name = argv[1];
	}

	page_size = sysconf(_SC_PAGE_SIZE);

	printf("open pathname:%s, page_size:%lu \n", proc_name, page_size);
	fflush(NULL);

	fd = open(proc_name, O_RDWR | O_SYNC);                                                                      
	if (fd < 0) {
		perror("open");
        return -1;
	}

    mapped_size = page_size * MAX_ZLOG_MEM_ORDER;
	printf("mapped_size: %u\n", mapped_size);

	addr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED) {
		perror("mmap");
        goto OUT;
	}

    /*
    paddr = 0x0;
	virt_to_phys_user(&paddr, getpid(), (uintptr_t)addr);
	printf("addr=%p, paddr1=0x%x\n", addr, (uintmax_t)paddr);
    */

	printf("addr=%p \n", addr);
	zring = (zlog_ring_t*)addr;

	printf("magic: 0x%x \n", zring->magic);
	printf("ver: 0x%x \n", zring->ver);
	printf("mem_size: %d \n", zring->ring_mem_size);
	printf("slot_cnt: %d \n", zring->slot_cnt);
	printf("slot_size: %d, %lu \n", zring->slot_size, sizeof(zlog_t));
	fflush(NULL);

    if (mapped_size != zring->ring_mem_size) {
        printf("mismatch buffer len: %u != %u \n", mapped_size, zring->ring_mem_size);
        goto OUT;
    }

	//memset(zring->data, 0, zring->cnt * zring->slot_size);

    if (0) {
        printf("wait 5 sec\n");
        fflush(NULL);
        sleep(1);
    }

	memset(&zread, 0, sizeof(zlog_read_t));

    srand( time(NULL) );
    zread.owner = rand() % 1024;
	printf("owner id=%d \n", zread.owner);

	len = read(fd, (void*)&zread, sizeof(zlog_read_t));

	printf("after owner id=%d \n", zread.owner);
	printf("read len=%d \n", len);
	printf("start=%d, cnt=%d \n", zread.start, zread.cnt);
    fflush(NULL);

	for (i=0; i<zread.cnt; i++) {
        char* base;
        zlog_t *log;

        uint32_t idx = zread.start + i;
        base = zlog_get_slot_addr(zring, idx);
        log = (zlog_t*)base;

		printf("log data[%d]: owner=%d,%s \n", idx, log->owner, log->msg);
	}

	//XXX:  move tail
	zring->tail = (zring->tail + zread.cnt) % zring->slot_cnt;

OUT:
	if (addr != MAP_FAILED) {
        munmap(addr, mapped_size);
	}

    if (fd != -1) {
        close(fd);
    }

	return EXIT_SUCCESS;
} 
 

