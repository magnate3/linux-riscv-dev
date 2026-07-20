#include "hugepage_memory.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>//isspace()
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>
#include "common.h"

//uint64_t virt_to_phys(void * addr);
//void * try_virt_area(size_t *size, size_t hugepage_sz);
//int find_numasocket(hugepage_file *hpf, uint32_t number);
char hugepage_size_str[256]="";
hugepage_memseg global_memseg[MAX_MEMSEG];
uint32_t nb_memsegs;

//find the the socket id of hugepage files
int find_numasocket(hugepage_file *hpf, uint32_t number)
{
	int socket_id;
	char *end, *nodestr;
	unsigned int i=0, hp_count=0;
	uint64_t virt_addr=0;
	char buf[BUFSIZ];
	FILE *f;
	
	f = fopen("/proc/self/numa_maps", "r");
	if(f == NULL){
		perror("Unable to open numa_maps file...:");
		return 0;
	}

	while(fgets(buf,sizeof(buf),f) != NULL){
		if(strstr(buf,"huge") == NULL && strstr(buf, "/mnt/hugepages/xk_map") ==NULL)
			continue;
		
		virt_addr = strtoull(buf, &end, 16);
		if(virt_addr == 0 || buf == end){
			printf("Can not find virtaddr...\n");
			goto ERROR;
		}

		//find string like " N0=1", 0 is numa node(socket) id, 1 is pages number
		nodestr = strstr(buf, " N");
		if(nodestr == NULL){
			printf("Can not find node information...\n");
			goto ERROR;
		}
		
		nodestr += 2;
		end = strstr(nodestr, "=");
		if(end == NULL){
			printf("Can not find node id...\n");
			goto ERROR;
		}
		end[0]='\0';
		end = NULL;

		socket_id = strtoul(nodestr, &end, 0);
		if( (nodestr[0]=='\0') || (end == NULL) || (*end != '\0')){
			printf("Can not find node id...\n");
			goto ERROR;
		}


		//find corresponding hugepage file and set value
		for(i=0; i<number; i++){
			if(hpf[i].addr == (void *)(unsigned long)virt_addr){
				hpf[i].socket_id = socket_id;
				hp_count++;
			}
		}
	}
	if(hp_count < number){
		goto ERROR;
	}
	fclose(f);
	return hp_count;

ERROR:
	fclose(f);
	return 0;	
}

//find a virt addr space with length of given size
void * try_virt_area(size_t *size, size_t hugepage_sz)
{
	void *addr=0;
	int fd;
	long aligned_addr;

	fd = open("/dev/zero", O_RDONLY);
	if(fd < 0){
		perror("Can not open dev zero:");
		return NULL;
	}
	do{
		addr = mmap(addr, (*size)+hugepage_sz, PROT_READ, MAP_PRIVATE, fd, 0);
		if(addr == MAP_FAILED)
			*size -= hugepage_sz;
	}while(addr == MAP_FAILED && *size > 0);
	
	if(addr == MAP_FAILED){
		close(fd);
		return NULL;
	}
	munmap(addr, (*size)+hugepage_sz);
	close(fd);

	//always move addr to next page start point
	//that's why the length of area we try to mmap is (*size)+hugepage_sz, 
	aligned_addr = (long)addr;
	addr = (void *)ALIGN_ADDR(aligned_addr, hugepage_sz);

	return addr;
}

//map virtaddr to physaddr
uint64_t virt_to_phys(void * addr)
{
	int page_size;
    unsigned long virt_pfn;
	off_t offset;
	int ret;
	uint64_t physaddr, page;
	int fd_pgmap;
			
	page_size = getpagesize();
	fd_pgmap = open("/proc/self/pagemap", O_RDONLY);//this file is a list of 64 bits' element, each element represent a page value
	if(fd_pgmap < 0 ){
		perror("Can not open pagemap file:");
		return 0;
	}
		
	virt_pfn = (unsigned long)addr / page_size;//virt_pfn is page frame number, that is the index of the page value for addr 
	offset = sizeof(uint64_t) * virt_pfn;//sizeof(uint64_t) is the size of element in file "/proc/self/pagemap", so index multiple size is the start of wanted element
	if(lseek(fd_pgmap, offset, SEEK_SET) == (off_t) - 1){//read the page value
		perror("seek error in pagemap:");
		goto EXIT;
	}	
	ret = read(fd_pgmap, &page, 8);
	if(ret<0){
		perror("Read pagemap error:");
		goto EXIT;
	}
	else if(ret != 8){
		printf("Read %d but expected 8...\n",ret);
		goto EXIT;
	}
	close(fd_pgmap);
	physaddr = ((page & 0x7fffffffffffffULL) << 12) + ((unsigned long)addr % page_size);
	
	return physaddr;
EXIT:
	close(fd_pgmap);
	return 0;	
}

//get current system's hugepage size
uint64_t get_cur_hugepgsz()
{
	FILE * fd = fopen("/proc/meminfo", "r");
	char buf[256];
	const char hugepagesz_str[]="Hugepagesize:";
	int i=0,j=0;
	uint64_t size = 0;
	int pgsz_str_len = strlen(hugepagesz_str);
	
	if(fd == NULL){
		perror("Can not open meminfo:");
		return -1;
	}
	while(fgets(buf, sizeof(buf), fd)){
		if(strncmp(buf, hugepagesz_str, pgsz_str_len) == 0){
			i = pgsz_str_len;
			while( isspace( (int)buf[i] ) )
				i++;
			while( buf[i] >= '0' && buf[i] <= '9' ){
				size = size*10 + (int)(buf[i]-'0');
				i++;
			}
			if(buf[i]==' ')
				i++;
			sprintf(hugepage_size_str, "hugepages-%lu%c%c",size,buf[i],buf[i+1]);
			switch(buf[i]){
				case 'G':
				case 'g': size *= 1024;
				case 'M':
				case 'm': size *= 1024;
				case 'K':
				case 'k': size *= 1024;
				default:
					break;
			}
		}	
	}
	return size;
}

//get current avaliable hugepages
uint32_t get_num_hugepages()
{
	char resv_path[PATH_MAX]="";
	char num_path[PATH_MAX]="";
	long unsigned resv_pages, num_pages;	

	if(strcmp(hugepage_size_str," ") == 0){
		printf("No valid hugepage size...\n");
		return -1;
	}
	sprintf(resv_path,"/sys/kernel/mm/hugepages/%s/resv_hugepages", hugepage_size_str);
	sprintf(num_path,"/sys/kernel/mm/hugepages/%s/free_hugepages", hugepage_size_str);
    if(parse_sysfs_value(resv_path, &resv_pages) != 0)
		return 0;	
    if(parse_sysfs_value(num_path, &num_pages) != 0)
		return 0;	
	if(num_pages == 0){
		printf("No free hugepages...\n");
		return 0;
	}
	if(num_pages >= resv_pages)
		num_pages -= resv_pages;
	else if(resv_pages != 0)
		num_pages = 0;
	if(num_pages > UINT32_MAX)
		num_pages = UINT32_MAX;
		
	return num_pages;	
}

static int cmp_physaddr(const void *p1, const void *p2)
{	
	hugepage_file *a = (hugepage_file*)p1;
	hugepage_file *b = (hugepage_file*)p2;
	
	if(a->physaddr < b->physaddr)
		return -1;
	else if(a->physaddr > b->physaddr)
		return 1;
	return 0;
}

//map hugepage file into userspace
uint32_t map_hugepages(hugepage_file *hpf, uint32_t number, uint64_t hugepage_sz)
{
	uint32_t i, j, conti_pages;
	int fd;
	void * virtaddr;
	void * vma_addr = NULL;
	int memseg_num=0;
	size_t vma_len = 0;	

//first map	
	for(i=0; i<number; i++){
		hpf[i].file_id = i;
		hpf[i].pagesize = hugepage_sz;
		memset(hpf[i].file_path, 0, sizeof(hpf[i].file_path));
		sprintf(hpf[i].file_path, "/mnt/hugepages/xk_map%u", hpf[i].file_id);
		hpf[i].file_path[sizeof(hpf[i].file_path)-1] = '\0';
		fd = open(hpf[i].file_path, O_CREAT | O_RDWR, 0600);
		if(fd<0){
			perror("hugepage file open error:");
			return i;
		}
		virtaddr = mmap(vma_addr, hugepage_sz, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);	
		if(virtaddr == MAP_FAILED){
			perror("Map hugepage file error:");
			return i;
		}
		hpf[i].addr = virtaddr;
		hpf[i].physaddr = virt_to_phys(hpf[i].addr);
		if(hpf[i].physaddr == 0){
			printf("Fail to get physaddr of %u page...\n", i);
			return i;
		}
		close(fd);	
	}

//sort by physaddr on increased order
	qsort(hpf, number, sizeof(hugepage_file), cmp_physaddr);

//find numa_socket
	if(find_numasocket(hpf, number) == 0){
		printf("Numa socket id set error...\n");
		return 0;
	}

//second map, try to get contiguous virtaddr
	for(i=0; i<number; i++){
		//get the length of a new pages area that have contiguous physaddr
		//then mmap from start addr of this area
		if(vma_len==0){
			for(j=i+1; j<number; j++){
				if((hpf[j].physaddr-hpf[j-1].physaddr) != hugepage_sz)
					break;
			}
			conti_pages = j-i;
			vma_len = conti_pages * hugepage_sz;
			//try get virt addr as long lengh as phyaddr
			vma_addr = try_virt_area(&vma_len, hugepage_sz);
			if(vma_addr == NULL)
				vma_len = hugepage_sz;
		}
		fd = open(hpf[i].file_path, O_CREAT | O_RDWR, 0600);
		if(fd<0){
			perror("hugepage file open error:");
			return i;
		}
		virtaddr = mmap(vma_addr, hugepage_sz, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);	
		if(virtaddr == MAP_FAILED){
			perror("Map hugepage file error:");
			return i;
		}
		
		close(fd);
		munmap(hpf[i].addr, hugepage_sz);
		hpf[i].addr = NULL;
		hpf[i].addr = virtaddr;
	    vma_len -= hugepage_sz;
		vma_addr = (char *)vma_addr + hugepage_sz;	
	}

	return i;
}

//clean dir
int clean_hugepages(const char * huge_dir){
	DIR *dir;
	struct dirent *dirent;
	int dir_fd, fd;
	const char filter[] = "*xk_map*";
	int lock_ret;	

	dir = opendir(huge_dir);
	if(!dir){
		perror("Unable to open hugepage dir:");
		return 1;
	}
	dir_fd = dirfd(dir);
	
	dirent = readdir(dir);
	if(!dirent){
		perror("Unable to read hugepage dir:");
		return 1;
	}	
	while(dirent != NULL){
		if( fnmatch(filter, dirent->d_name, 0) > 0){
			dirent = readdir(dir);
			continue;
		}
		//try and lock the file
		fd = openat(dir_fd, dirent->d_name, O_RDONLY);
		
		//go to next file if fail
		if(fd == -1){
			dirent = readdir(dir);
			continue;
		}
		
        //try non_block lock
		lock_ret = flock(fd, LOCK_EX|LOCK_NB);
		
		if(lock_ret != -1){
			//unlock and remove it
			flock(fd, LOCK_UN);
			unlinkat(dir_fd, dirent->d_name, 0);
		}
		close(fd);
		dirent = readdir(dir);			
	}
	return 0;
}

//unmap all hugepages file
uint32_t munmap_all_hugepages(hugepage_file *hpf, uint32_t page_number)
{
	uint32_t i=0;
	for(i=0; i<page_number; i++){
		if(hpf[i].addr != NULL && hpf[i].pagesize > 0)
			munmap(hpf[i].addr, hpf[i].pagesize);
	}
}

//merge pages to segments
uint32_t pages_to_memsegs(hugepage_file *hpf, uint32_t page_number)
{
	uint32_t i=0, j=-1;
	bool is_new_memseg;
	
	for(i=0; i<page_number; i++){
		is_new_memseg = 0;
		if(i==0)//first pages start a new memseg
			is_new_memseg = 1;
		else if(hpf[i].socket_id != hpf[i-1].socket_id)//different socket id
			is_new_memseg = 1;
		else if(hpf[i].pagesize != hpf[i-1].pagesize)//different page size
			is_new_memseg = 1;
		else if( (hpf[i].physaddr - hpf[i-1].physaddr) != hpf[i].pagesize)//uncontiguous physaddr
			is_new_memseg = 1;
		else if( ((unsigned long)hpf[i].addr - (unsigned long)hpf[i-1].addr) != hpf[i].pagesize)//uncontiguous virt addr
			is_new_memseg = 1;
		
		if(is_new_memseg){
			j++;
			global_memseg[j].phys_addr = hpf[i].physaddr;
			global_memseg[j].addr = hpf[i].addr;
			global_memseg[j].len = hpf[i].pagesize;
			global_memseg[j].socket_id = hpf[i].socket_id;
			global_memseg[j].hugepage_sz = hpf[i].pagesize; 
			hpf[i].memseg_id = j;
			//j ++;//if j ++ here, if next page is in the same memseg in the next loop,
				   //the (j+1)th memseg's len will be added by pagesize instead of jth memsegs
		}
		else{
			global_memseg[j].len += hpf[i].pagesize;
			hpf[i].memseg_id = j;
		}
	}
	if(i < page_number){
		printf("Can not merge all hugepages...\n");
		munmap_all_hugepages(hpf, page_number);
	}
	nb_memsegs = j+1;
	return i;
}

//print memsegse state
void show_memsegs_state()
{
	uint32_t i=0;
	size_t total_size=0;

	for(i=0; i<nb_memsegs; i++)
	{
		printf("%uth memseg size=%lu\n",i, global_memseg[i].len);
		total_size += global_memseg[i].len;
	}
	printf("Total size:%lu\n",total_size);
	return;
}
