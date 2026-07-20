#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <stdint.h>
#include <linux/kvm.h>
#include<inttypes.h>
#include"filesystem.h"

/* CR0 bits */
#define CR0_PE 1u
#define CR0_MP (1U << 1)
#define CR0_EM (1U << 2)
#define CR0_TS (1U << 3)
#define CR0_ET (1U << 4)
#define CR0_NE (1U << 5)
#define CR0_WP (1U << 16)
#define CR0_AM (1U << 18)
#define CR0_NW (1U << 29)
#define CR0_CD (1U << 30)
#define CR0_PG (1U << 31)

/* CR4 bits */
#define CR4_VME 1
#define CR4_PVI (1U << 1)
#define CR4_TSD (1U << 2)
#define CR4_DE (1U << 3)
#define CR4_PSE (1U << 4)
#define CR4_PAE (1U << 5)
#define CR4_MCE (1U << 6)
#define CR4_PGE (1U << 7)
#define CR4_PCE (1U << 8)
#define CR4_OSFXSR (1U << 8)
#define CR4_OSXMMEXCPT (1U << 10)
#define CR4_UMIP (1U << 11)
#define CR4_VMXE (1U << 13)
#define CR4_SMXE (1U << 14)
#define CR4_FSGSBASE (1U << 16)
#define CR4_PCIDE (1U << 17)
#define CR4_OSXSAVE (1U << 18)
#define CR4_SMEP (1U << 20)
#define CR4_SMAP (1U << 21)

#define EFER_SCE 1
#define EFER_LME (1U << 8)
#define EFER_LMA (1U << 10)
#define EFER_NXE (1U << 11)

/* 32-bit page directory entry bits */
#define PDE32_PRESENT 1
#define PDE32_RW (1U << 1)
#define PDE32_USER (1U << 2)
#define PDE32_PS (1U << 7)

/* 64-bit page * entry bits */
#define PDE64_PRESENT 1
#define PDE64_RW (1U << 1)
#define PDE64_USER (1U << 2)
#define PDE64_ACCESSED (1U << 5)
#define PDE64_DIRTY (1U << 6)
#define PDE64_PS (1U << 7)
#define PDE64_G (1U << 8)


struct vm {
	int sys_fd;
	int fd;
	char *mem;
};

struct vcpu {
	int fd;
	struct kvm_run *kvm_run;
};
struct fd_map{
	int fd[1024];
};

void vm_init(struct vm *vm, size_t mem_size)
{
	int api_ver;
	struct kvm_userspace_memory_region memreg;
	//printf("Getting KVM DEV FD\n");
	vm->sys_fd = open("/dev/kvm", O_RDWR);
	if (vm->sys_fd < 0) {
		perror("open /dev/kvm");
		exit(1);
	}
	//printf("Checking Api Version..\n");
	api_ver = ioctl(vm->sys_fd, KVM_GET_API_VERSION, 0);
	if (api_ver < 0) {
		perror("KVM_GET_API_VERSION\n");
		exit(1);
	}

	if (api_ver != KVM_API_VERSION) {
		fprintf(stderr, "Got KVM api version %d, expected %d\n",
			api_ver, KVM_API_VERSION);
		exit(1);
	}
	//printf("Create VM on KVM FD\n");
	vm->fd = ioctl(vm->sys_fd, KVM_CREATE_VM, 0);
	if (vm->fd < 0) {
		perror("KVM_CREATE_VM");
		exit(1);
	}

        if (ioctl(vm->fd, KVM_SET_TSS_ADDR, 0xfffbd000) < 0) {
                perror("KVM_SET_TSS_ADDR");
		exit(1);
	}
	//printf("Creating Virtual machine memory of size %ld\n",mem_size);
	vm->mem = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (vm->mem == MAP_FAILED) {
		perror("mmap mem");
		exit(1);
	}

	madvise(vm->mem, mem_size, MADV_MERGEABLE);
	//printf("Setting Kvm User Space Memory Region Need to read this\n");
	memreg.slot = 0;
	memreg.flags = 0;
	memreg.guest_phys_addr = 0;
	memreg.memory_size = mem_size;
	memreg.userspace_addr = (unsigned long)vm->mem;
        if (ioctl(vm->fd, KVM_SET_USER_MEMORY_REGION, &memreg) < 0) {
		perror("KVM_SET_USER_MEMORY_REGION");
                exit(1);
	}
}

void vcpu_init(struct vm *vm, struct vcpu *vcpu)
{
	int vcpu_mmap_size;
	//printf("creating VCPU");
	vcpu->fd = ioctl(vm->fd, KVM_CREATE_VCPU, 0);
        if (vcpu->fd < 0) {
		perror("KVM_CREATE_VCPU");
                exit(1);
	}
	//printf("getting VCPU kvm_run size \n");
	vcpu_mmap_size = ioctl(vm->sys_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
        if (vcpu_mmap_size <= 0) {
		perror("KVM_GET_VCPU_MMAP_SIZE");
                exit(1);
	}
	//printf("KVM Run Size %d memory is allocated using mmmap\n",vcpu_mmap_size);
	vcpu->kvm_run = mmap(NULL, vcpu_mmap_size, PROT_READ | PROT_WRITE,
			     MAP_SHARED, vcpu->fd, 0);
	if (vcpu->kvm_run == MAP_FAILED) {
		perror("mmap kvm_run");
		exit(1);
	}
}
void init_fd_map(struct fd_map *map){
	for(int i=0;i<1024;i++){
		map->fd[i]=-1;
	}
	map->fd[0]=0;
	map->fd[1]=1;
	map->fd[2]=2;
}
int run_vm(struct vm *vm, struct vcpu *vcpu, size_t sz)
{
	struct fd_map fdmap;
	init_fd_map(&fdmap);
	int nExit=0;
	struct kvm_regs regs;
	uint64_t memval = 0;
	for (;;) {
		if (ioctl(vcpu->fd, KVM_RUN, 0) < 0) {
			perror("KVM_RUN");
			exit(1);
		}
		nExit++;
		switch (vcpu->kvm_run->exit_reason) {
			case KVM_EXIT_HLT:
				goto check;
			//case KVM_EXIT_MMIO:
			//        printf("KVM_EXIT_MMIO\n");
			//	printf("is_write %d",vcpu->kvm_run->mmio.is_write);
			//	printf("data %hhn",vcpu->kvm_run->mmio.data);
			//	printf("phys addr %lld",vcpu->kvm_run->mmio.phys_addr);
			//	printf("len %d",vcpu->kvm_run->mmio.len);
			//	continue;
		case KVM_EXIT_IO:
			//printf("KVM_EXIT_IO\n");
			if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT
			    && vcpu->kvm_run->io.port == CHAR_PORT) {
				char *char_pointer= (char *)vcpu->kvm_run+vcpu->kvm_run->io.data_offset;
				fwrite(char_pointer,vcpu->kvm_run->io.size, 1, stdout);
				fflush(stdout);
				continue;
			}
            if(vcpu->kvm_run->io.port & HYPERCALL) {				
			        //printf("KVM_EXIT_HYPERCALL\n");
				switch (vcpu->kvm_run->io.port)
				{
				case HC_PRINT_STRING:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						char* char_pointer= (char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset;
						uint32_t offset=*((uint32_t*)char_pointer);
						char_pointer=vm->mem+offset;
						printf("%s",char_pointer);
						fflush(stdout);
					}
					continue;
				case HC_PRINT_VALUE:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						char *char_pointer= (char *)vcpu->kvm_run+vcpu->kvm_run->io.data_offset;
			    		printf("%" PRIu32,*((uint32_t*)char_pointer));
						fflush(stdout);
					}
					continue;
				case HC_NUM_EXIT:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_IN){
						*(uint32_t*)((	char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset)=nExit;
					}
					continue;
				case HC_OPEN:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						uint32_t offset=*(uint32_t*)((char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset);
						char *char_pointer=vm->mem+offset;
						struct open_data *data=(struct open_data*)char_pointer;
						char *filename=vm->mem+data->fileoffset;
						int fd_generated=open(filename,data->flags,S_IRWXG);
						if(fd_generated!=-1){
							int i=3;
							for(;i<1024;i++){
								if(fdmap.fd[i]==-1){
									break;
								}
							}
							if(i==1024){
								printf("GUEST OS HAS MAX FDS");
								data->fd=-1;
							}
							else{
								data->fd=i;
								fdmap.fd[i]=fd_generated;
							}
						}
						data->errno_hc=errno;
						printf("FD GENERATED IN HYPERVISOR %d,FD SENT TO GUEST %d FILE NAME %s AND FLAGS %d\n",fd_generated,data->fd,filename,data->flags);
					}
					continue;
				case HC_CLOSE:;
				    if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						uint32_t offset=*(uint32_t*)((char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset);
						char *char_pointer=vm->mem+offset;
						struct close_data *data=(struct close_data*)char_pointer;
						if(data->close_fd>1024){
							data->result=-1;
							data->errno_hc=EBADF;
						}else{
						data->result=close(fdmap.fd[data->close_fd]);
						data->errno_hc=errno;
						fdmap.fd[data->close_fd]=-1;
						}
						printf("CLOSE FD %d and CLOSE STATUS %d\n",data->close_fd,data->result);	
					}
					continue;
				case HC_READ:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						uint32_t offset=*(uint32_t*)((char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset);
						char *char_pointer=vm->mem+offset;
						struct read_write_data *data=(struct read_write_data*)char_pointer;
						char *buff=vm->mem+data->buffer_offset;
						if(data->fd>1024){
							data->result=-1;
							data->errno_hc=EBADF;
						}else{
						data->result=read(fdmap.fd[data->fd],buff,data->count);
						data->errno_hc=errno;
						}
						printf("READ FD  %d,HYPERVISOR MAPED FD %d SIZE %ld AND READ RETURN VALUE %ld\n",data->fd,fdmap.fd[data->fd],data->count,data->result);
					}
					continue;
				
				case HC_WRITE:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						uint32_t offset=*(uint32_t*)((char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset);
						char *char_pointer=vm->mem+offset;
						struct read_write_data *data=(struct read_write_data*)char_pointer;
						char *buff=vm->mem+data->buffer_offset;
						if(data->fd>1024){
							data->result=-1;
							data->errno_hc=EBADF;
						}else{
						data->result=write(fdmap.fd[data->fd],buff,data->count);
						fsync(fdmap.fd[data->fd]);
						data->errno_hc=errno;
						}
						printf("WRITE FD  %d,HYPERVISOR MAPPED FD %d SIZE %ld AND READ RETURN VALUE %ld\n",data->fd,fdmap.fd[data->fd],data->count,data->result);
					}
					continue;
				case HC_LSEEK:;
					if(vcpu->kvm_run->io.direction==KVM_EXIT_IO_OUT){
						uint32_t offset=*(uint32_t*)((char*)vcpu->kvm_run+vcpu->kvm_run->io.data_offset);
						char *char_pointer=vm->mem+offset;
						struct lseek_data *data=(struct lseek_data*)char_pointer;
						if(data->fd>1024){
							data->result=-1;
							data->errno_hc=EBADF;
						}else{
							data->result=lseek(fdmap.fd[data->fd],data->offset,data->whence);
							data->errno_hc=errno;
						}
						printf("LSEEK FD %d,HYPERVISOR MAPPED FD %d offset %ld AND WHENCE %d\n",data->fd,fdmap.fd[data->fd],data->offset,data->whence);
					}
					
					continue;

				default:
					printf("Invalid hypercall port %d direction %d\n ",vcpu->kvm_run->io.port,vcpu->kvm_run->io.direction);
					goto check;
				}
			}
			
			/* fall through */
		default:
			fprintf(stderr,	"Got exit_reason %d,"
				" expected KVM_EXIT_HLT (%d) %d\n",
				vcpu->kvm_run->exit_reason, KVM_EXIT_HLT,KVM_EXIT_MMIO);
			exit(1);
		}
	}

 check:
	if (ioctl(vcpu->fd, KVM_GET_REGS, &regs) < 0) {
		perror("KVM_GET_REGS");
		exit(1);
	}
	if (regs.rax != 42) {
		printf("Wrong result: {E,R,}AX is %lld\n", regs.rax);
		return 0;
	}

	memcpy(&memval, &vm->mem[0x400], sz);
	if (memval != 42) {
		printf("Wrong result: memory at 0x400 is %lld\n",
		       (unsigned long long)memval);
		return 0;
	}

	return 1;
}

extern const unsigned char guest16[], guest16_end[];

int run_real_mode(struct vm *vm, struct vcpu *vcpu)
{
	struct kvm_sregs sregs;
	struct kvm_regs regs;

	printf("Testing real mode\n");

        if (ioctl(vcpu->fd, KVM_GET_SREGS, &sregs) < 0) {
		perror("KVM_GET_SREGS");
		exit(1);
	}

	sregs.cs.selector = 0;
	sregs.cs.base = 0;

        if (ioctl(vcpu->fd, KVM_SET_SREGS, &sregs) < 0) {
		perror("KVM_SET_SREGS");
		exit(1);
	}

	memset(&regs, 0, sizeof(regs));
	/* Clear all FLAGS bits, except bit 1 which is always set. */
	regs.rflags = 2;
	regs.rip = 0;

	if (ioctl(vcpu->fd, KVM_SET_REGS, &regs) < 0) {
		perror("KVM_SET_REGS");
		exit(1);
	}

	memcpy(vm->mem, guest16, guest16_end-guest16);
	return run_vm(vm, vcpu, 2);
}

static void setup_protected_mode(struct kvm_sregs *sregs)
{
	struct kvm_segment seg = {
		.base = 0,//base is set to zero
		.limit = 0xffffffff,
		.selector = 1 << 3,
		.present = 1,
		.type = 11, /* Code: execute, read, accessed */
		.dpl = 0,/*descriptor privellege level ring0 dpl=0 ring3 dpl=3*/
		.db = 1,
		.s = 1, /* Code/data */
		.l = 0,
		.g = 1, /* 4KB granularity */
	};

	sregs->cr0 |= CR0_PE; /* enter protected mode */

	sregs->cs = seg;

	seg.type = 3; /* Data: read/write, accessed */
	seg.selector = 2 << 3;
	sregs->ds = sregs->es = sregs->fs = sregs->gs = sregs->ss = seg;
}

extern const unsigned char guest32[], guest32_end[];

int run_protected_mode(struct vm *vm, struct vcpu *vcpu)
{
	struct kvm_sregs sregs;
	struct kvm_regs regs;

	printf("Testing protected mode\n");

        if (ioctl(vcpu->fd, KVM_GET_SREGS, &sregs) < 0) {
		perror("KVM_GET_SREGS");
		exit(1);
	}

	setup_protected_mode(&sregs);

        if (ioctl(vcpu->fd, KVM_SET_SREGS, &sregs) < 0) {
		perror("KVM_SET_SREGS");
		exit(1);
	}

	memset(&regs, 0, sizeof(regs));
	/* Clear all FLAGS bits, except bit 1 which is always set. */
	regs.rflags = 2;
	regs.rip = 0;

	if (ioctl(vcpu->fd, KVM_SET_REGS, &regs) < 0) {
		perror("KVM_SET_REGS");
		exit(1);
	}

	memcpy(vm->mem, guest32, guest32_end-guest32);
	return run_vm(vm, vcpu, 4);
}

static void setup_paged_32bit_mode(struct vm *vm, struct kvm_sregs *sregs)
{
	uint32_t pd_addr = 0x2000;
	uint32_t *pd = (void *)(vm->mem + pd_addr);

	/* A single 4MB page to cover the memory region */
	pd[0] = PDE32_PRESENT | PDE32_RW | PDE32_USER | PDE32_PS;
	/* Other PDEs are left zeroed, meaning not present. */

	sregs->cr3 = pd_addr;
	sregs->cr4 = CR4_PSE;
	sregs->cr0
		= CR0_PE | CR0_MP | CR0_ET | CR0_NE | CR0_WP | CR0_AM | CR0_PG;
	sregs->efer = 0;
}

int run_paged_32bit_mode(struct vm *vm, struct vcpu *vcpu)
{
	struct kvm_sregs sregs;
	struct kvm_regs regs;

	printf("Testing 32-bit paging\n");

        if (ioctl(vcpu->fd, KVM_GET_SREGS, &sregs) < 0) {
		perror("KVM_GET_SREGS");
		exit(1);
	}

	setup_protected_mode(&sregs);
	setup_paged_32bit_mode(vm, &sregs);

        if (ioctl(vcpu->fd, KVM_SET_SREGS, &sregs) < 0) {
		perror("KVM_SET_SREGS");
		exit(1);
	}

	memset(&regs, 0, sizeof(regs));
	/* Clear all FLAGS bits, except bit 1 which is always set. */
	regs.rflags = 2;
	regs.rip = 0;

	if (ioctl(vcpu->fd, KVM_SET_REGS, &regs) < 0) {
		perror("KVM_SET_REGS");
		exit(1);
	}

	memcpy(vm->mem, guest32, guest32_end-guest32);
	return run_vm(vm, vcpu, 4);
}

extern const unsigned char guest64[], guest64_end[];

static void setup_64bit_code_segment(struct kvm_sregs *sregs)
{
	struct kvm_segment seg = {
		.base = 0,
		.limit = 0xffffffff,
		.selector = 1 << 3,
		.present = 1,
		.type = 11, /* Code: execute, read, accessed */
		.dpl = 0,
		.db = 0,
		.s = 1, /* Code/data */
		.l = 1,
		.g = 1, /* 4KB granularity */
	};

	sregs->cs = seg;

	seg.type = 3; /* Data: read/write, accessed */
	seg.selector = 2 << 3;
	sregs->ds = sregs->es = sregs->fs = sregs->gs = sregs->ss = seg;
}

static void setup_long_mode(struct vm *vm, struct kvm_sregs *sregs)
{
	uint64_t pml4_addr = 0x2000;
	uint64_t *pml4 = (void *)(vm->mem + pml4_addr);

	uint64_t pdpt_addr = 0x3000;
	uint64_t *pdpt = (void *)(vm->mem + pdpt_addr);

	uint64_t pd_addr = 0x4000;
	uint64_t *pd = (void *)(vm->mem + pd_addr);

	pml4[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pdpt_addr;
	pdpt[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pd_addr;
	pd[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | PDE64_PS;

	sregs->cr3 = pml4_addr;
	sregs->cr4 = CR4_PAE;//it tells that extended adderessing enable
	sregs->cr0
		= CR0_PE | CR0_MP | CR0_ET | CR0_NE | CR0_WP | CR0_AM | CR0_PG;
	sregs->efer = EFER_LME | EFER_LMA;//these enable long mode LME long mode enable
									//lma long mode active
									//to enable syscall we have to set this register to
									//regs->efer=EFER_SCE or value 0x1

	setup_64bit_code_segment(sregs);
}

int run_long_mode(struct vm *vm, struct vcpu *vcpu)
{
	struct kvm_sregs sregs;
	struct kvm_regs regs;

	printf("Testing 64-bit mode\n");

        if (ioctl(vcpu->fd, KVM_GET_SREGS, &sregs) < 0) {
		perror("KVM_GET_SREGS");
		exit(1);
	}
	//printf("Setting long mode special registers\n");
	setup_long_mode(vm, &sregs);

        if (ioctl(vcpu->fd, KVM_SET_SREGS, &sregs) < 0) {
		perror("KVM_SET_SREGS");
		exit(1);
	}

	memset(&regs, 0, sizeof(regs));
	/* Clear all FLAGS bits, except bit 1 which is always set. */
	regs.rflags = 2;
	regs.rip = 0;
	/* Create stack at top of 2 MB page and grow down. */
	regs.rsp = 2 << 20;

	if (ioctl(vcpu->fd, KVM_SET_REGS, &regs) < 0) {
		perror("KVM_SET_REGS");
		exit(1);
	}

	memcpy(vm->mem, guest64, guest64_end-guest64);
	return run_vm(vm, vcpu, 8);
}


int main(int argc, char **argv)
{
	struct vm vm;
	struct vcpu vcpu;
	enum {
		REAL_MODE,
		PROTECTED_MODE,
		PAGED_32BIT_MODE,
		LONG_MODE,
	} mode = LONG_MODE;
	int opt;

	while ((opt = getopt(argc, argv, "rspl")) != -1) {
		switch (opt) {
		case 'r':
			mode = REAL_MODE;
			break;

		case 's':
			mode = PROTECTED_MODE;
			break;

		case 'p':
			mode = PAGED_32BIT_MODE;
			break;

		case 'l':
			mode = LONG_MODE;
			break;

		default:
			fprintf(stderr, "Usage: %s [ -r | -s | -p | -l ]\n",
				argv[0]);
			return 1;
		}
	}

	vm_init(&vm, 0x200000);
	vcpu_init(&vm, &vcpu);

	switch (mode) {
	case REAL_MODE:
		return !run_real_mode(&vm, &vcpu);

	case PROTECTED_MODE:
		return !run_protected_mode(&vm, &vcpu);

	case PAGED_32BIT_MODE:
		return !run_paged_32bit_mode(&vm, &vcpu);

	case LONG_MODE:
		return !run_long_mode(&vm, &vcpu);
	}

	return 1;
}
