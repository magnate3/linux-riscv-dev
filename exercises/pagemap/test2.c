#define _GNU_SOURCE 1
#define __USE_GNU 

#include<stdio.h> 
#include<unistd.h>
#include<string.h>
#include<stdlib.h>
#include <sched.h>
#include <math.h>
#include <numa.h>
#include <stdint.h>
#include <fcntl.h>


#define phys_addr_t     u_int64_t
#define PFN_MASK_SIZE   8

 
void executeCMD(const char *cmd, char *result)
{
    char buf_ps[1024];
    char ps[1024]={0};
    FILE *ptr;
    strcpy(ps, cmd);
    if((ptr=popen(ps, "r"))!=NULL)
    {
        while(fgets(buf_ps, 1024, ptr)!=NULL)
        {
//	       可以通过这行来获取shell命令行中的每一行的输出
//	   	   printf("%s", buf_ps);
           strcat(result, buf_ps);
           if(strlen(result)>1024)
               break;
        }
        pclose(ptr);
        ptr = NULL;
    }
    else
    {
        printf("popen %s error\n", ps);
    }
}

/*Find va->pa translation */
phys_addr_t
rte_mem_virt2phy(const void *virtaddr)
{
        int fd, retval;
        uint64_t page, physaddr;
        unsigned long virt_pfn;
        int page_size;
        off_t offset;

        /* standard page size */
        page_size = getpagesize();
        // printf("page_size:%d \n", page_size);

        char result[1024]={0};
        executeCMD("ls -l /proc/self/pagemap", result);
        printf("%s",result);

        fd = open("/proc/self/pagemap", O_RDONLY);
        // printf("fd:%ld\n", fd);
        if (fd < 0) { 
                printf("open /proc/self/pagemap error!\n");
        }

        virt_pfn = (unsigned long)virtaddr / page_size;
        offset = sizeof(uint64_t) * virt_pfn;
        if (lseek(fd, offset, SEEK_SET) == (off_t) -1) {
                printf("can't find corrsponding page in /proc/self/pagemap !\n");
                return -1;
        }

        retval = read(fd, &page, PFN_MASK_SIZE);
        printf("page:0x%lx\n\n", page);
        if(retval < 0){
            printf("read /proc/self/pagemap error!\n");
        }
        
        close(fd);

        /*
         * the pfn (page frame number) are bits 0-54 (see
         * pagemap.txt in linux Documentation)
         */
        if ((page & 0x7fffffffffffffULL) == 0)
                return -1;

        physaddr = ((page & 0x7fffffffffffffULL) * page_size)
                + ((unsigned long)virtaddr % page_size);

        return physaddr;
}



int main()
{
    size_t memsize = sizeof(int) * 1024 * 1024;
	int *mem0 = numa_alloc_onnode(memsize, 0);
	if (mem0 == NULL){
		perror("numa_alloc_onnode 0");
	}

	printf("success alloc on node0, vaddress: 0x%x.\n", mem0);

    *(mem0 + 4096) = 10;  //赋值后才会分配实地址
    // printf("value:%d\n", *(mem0 + 4096));

    unsigned long phyaddress0 = rte_mem_virt2phy(mem0 + 4096);
    printf("the physic address alloc in node0: 0x%lx \n", phyaddress0);

    int *mem1 = numa_alloc_onnode(memsize, 1);
	if (mem1 == NULL){
		perror("numa_alloc_onnode 1");
	}

    *(mem1 + 4096) = 10;

	printf("success alloc on node0, vaddress: 0x%x.\n", mem1);
    unsigned long phyaddress1 = rte_mem_virt2phy(mem1+ 4096);
    printf("the physic address alloc in node1: 0x%lx \n", phyaddress1);


	int *mem2 = numa_alloc_local(memsize);
	if (mem2 == NULL){
		perror("numa_alloc_local");
	}

    *(mem2 + 4096) = 10;

	printf("success alloc on local, vaddress: 0x%x.\n", mem2);
    unsigned long phyaddress2 = rte_mem_virt2phy(mem2 + 4096);
    printf("the physic address alloc local: 0x%lx \n", phyaddress2);


    numa_free(mem0, memsize); 
    numa_free(mem1, memsize);
	numa_free(mem2, memsize);
}