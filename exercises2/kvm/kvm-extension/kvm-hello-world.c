#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <stdint.h>
#include <linux/kvm.h>

#include "./common-host.h"

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

void vm_init(struct vm *vm, size_t mem_size)
{
	int api_ver;
	struct kvm_userspace_memory_region memreg;

	vm->sys_fd = open("/dev/kvm", O_RDWR);
	if (vm->sys_fd < 0) {
		perror("open /dev/kvm");
		exit(1);
	}

	api_ver = ioctl(vm->sys_fd, KVM_GET_API_VERSION, 0);
	if (api_ver < 0) {
		perror("KVM_GET_API_VERSION");
		exit(1);
	}

	if (api_ver != KVM_API_VERSION) {
		fprintf(stderr, "Got KVM api version %d, expected %d\n",
			api_ver, KVM_API_VERSION);
		exit(1);
	}
	printf("api version %d \n",api_ver);
	vm->fd = ioctl(vm->sys_fd, KVM_CREATE_VM, 0);
	if (vm->fd < 0) {
		perror("KVM_CREATE_VM");
		exit(1);
	}

	if (ioctl(vm->fd, KVM_SET_TSS_ADDR, 0xfffbd000) < 0) {
		perror("KVM_SET_TSS_ADDR");
		exit(1);
	}

	vm->mem = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (vm->mem == MAP_FAILED) {
		perror("mmap mem");
		exit(1);
	}
	// printf("Mmap address %p\n",vm->mem);

	madvise(vm->mem, mem_size, MADV_MERGEABLE);

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

struct vcpu {
	int fd;
	struct kvm_run *kvm_run;
};

void vcpu_init(struct vm *vm, struct vcpu *vcpu)
{
	int vcpu_mmap_size;

	vcpu->fd = ioctl(vm->fd, KVM_CREATE_VCPU, 0);
        if (vcpu->fd < 0) {
		perror("KVM_CREATE_VCPU");
                exit(1);
	}

	vcpu_mmap_size = ioctl(vm->sys_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
        if (vcpu_mmap_size <= 0) {
		perror("KVM_GET_VCPU_MMAP_SIZE");
                exit(1);
	}

	vcpu->kvm_run = mmap(NULL, vcpu_mmap_size, PROT_READ | PROT_WRITE,
			     MAP_SHARED, vcpu->fd, 0);
	if (vcpu->kvm_run == MAP_FAILED) {
		perror("mmap kvm_run");
		exit(1);
	}
}
int total_exits = 0;
int ret=-2;
char * read_data;
uintptr_t read_addr;

int run_vm(struct vm *vm, struct vcpu *vcpu, size_t sz)
{
	struct kvm_regs regs;
	uint64_t memval = 0;

	for (;;) {
		if (ioctl(vcpu->fd, KVM_RUN, 0) < 0) {
			perror("KVM_RUN");
			exit(1);
		}
		total_exits = total_exits + 1;
		// printf("Exit done %d total exits%d\n",vcpu->kvm_run->exit_reason, total_exits);
		switch (vcpu->kvm_run->exit_reason) {
		case KVM_EXIT_HLT:
			printf("KVM blocked KVM_EXIT_HLT %d \n",vcpu->kvm_run->exit_reason);
			goto check;

		case KVM_EXIT_IO:
			if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT
			    && vcpu->kvm_run->io.port == 0xE9) {
				char *p = (char *)vcpu->kvm_run;
				fwrite(p + vcpu->kvm_run->io.data_offset,
				       vcpu->kvm_run->io.size, 1, stdout);
				fflush(stdout);
				printf("KVM blocked KVM_EXIT_IO E9 %d \n",vcpu->kvm_run->exit_reason);
				continue;
			}

			if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT
			    && vcpu->kvm_run->io.port == 0xE5) {
				char *p = (char *)vcpu->kvm_run;
				uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
				uint32_t* data_address = (uint32_t*)(p+offset);
				
				// fwrite(p + vcpu->kvm_run->io.data_offset,
				//        vcpu->kvm_run->io.size, 1, stdout);
				// fflush(stdoust);

				printf("Printed %d \n",*(data_address));
				// printf("Offset %d IO size %d\n",offset, vcpu->kvm_run->io.size);
				// printf("KVM blocked KVM_EXIT_IO E5 %d \n",vcpu->kvm_run->exit_reason);
				continue;
			}
			if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT
			    && vcpu->kvm_run->io.port == PRINT_STRING) {
				char *p = (char *)vcpu->kvm_run;
				uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
				uint32_t data_address = *(uint32_t*)(p + offset);
				char * str = (char *) &vm->mem[data_address];
				// char * str = (char *)&data_address;
				printf("OP string -> %s\n",str);
				continue;
			}

			if(vcpu->kvm_run->io.port == OPEN){
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t data_address = *(uint32_t*)(p + offset);
					open_file_t* open_file_ptr = (open_file_t* ) &vm->mem[data_address];
					ret = open(open_file_ptr->data, O_CREAT | O_RDWR, 0644);
					printf("Open call -> %s fd is %d\n",open_file_ptr->data,ret);
					continue;
				}
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_IN ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t* data_address = (uint32_t*)(p+offset);
					uint32_t op =ret;
					memcpy( data_address, &op,sizeof(uint32_t));
					continue;
				}
			}
			if(vcpu->kvm_run->io.port == WRITE){
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t data_address = *(uint32_t*)(p + offset);
					write_file_t* write_file_ptr = (write_file_t* ) &vm->mem[data_address];
					// uintptr_t address_str =(uintptr_t) ((write_file_t* ) &vm->mem[data_address])->data;
					// printf("Write file address %ld\n",address_str);
					// write_file_ptr->data = (char *) malloc(404);
					// strcpy(write_file_ptr->data,(char *)&vm->mem[address_str+8]);
					// printf("Write file fd %d len is %d\n",write_file_ptr->fd,write_file_ptr->len);
					printf("Write file data from guest:- %s\n",write_file_ptr->data);
					ret = write(write_file_ptr->fd,write_file_ptr->data,write_file_ptr->len);
					// printf("Error number %d\n",errno);
					continue;
				}
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_IN ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t* data_address = (uint32_t*)(p+offset);
					uint32_t op =ret;
					memcpy( data_address, &op,sizeof(uint32_t));
					continue;
				}
			}


			if(vcpu->kvm_run->io.port == READ){
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t data_address = *(uint32_t*)(p + offset);

					read_file_t* read_file_ptr = (read_file_t* ) &vm->mem[data_address];
					// read_addr =(uintptr_t) ((read_file_t* ) &vm->mem[data_address])->data;

					// printf("Read file address %ld\n",read_addr);
					// read_file_ptr->data = (char *) malloc(404);
					// printf("Read file fd %d len is %d\n",read_file_ptr->fd,read_file_ptr->len);
					ret = read(read_file_ptr->fd,read_file_ptr->data,read_file_ptr->len);
					// printf("Read file %s ;ret is %d  \n",read_file_ptr->data,ret);
					// printf("Error in file %d\n", errno);
					// printf("Read addrress from guest %ld\n",read_addr);
					// memcpy((char *)((read_file_t*) &vm->mem[data_address])->data,read_file_ptr->data,ret);
					// printf("Data address %s\n",((read_file_t*)&vm->mem[data_address])->data);
					printf("Read file:- %s ret:%d\n",read_file_ptr->data,ret);
					continue;
				}
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_IN ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t* data_address = (uint32_t*)(p+offset);
					uint32_t op =ret;
					memcpy( data_address, &op,sizeof(uint32_t));
					continue;
				}
			}
			
			if(vcpu->kvm_run->io.port == SEEK){
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_OUT ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t data_address = *(uint32_t*)(p + offset);

					seek_file_t* seek_file_ptr = (seek_file_t* ) &vm->mem[data_address];
					
					ret = lseek(seek_file_ptr->fd,seek_file_ptr->pos,SEEK_SET);
					printf("File seek info ret %d errno %d\n", ret, errno);
					continue;
				}
				if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_IN ) {
					char *p = (char *)vcpu->kvm_run;
					uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
					uint32_t* data_address = (uint32_t*)(p+offset);
					uint32_t op =ret;
					memcpy( data_address, &op,sizeof(uint32_t));
					continue;
				}
			}

			if (vcpu->kvm_run->io.direction == KVM_EXIT_IO_IN
			    && vcpu->kvm_run->io.port == GET_EXITS) {
				char *p = (char *)vcpu->kvm_run;
				uint32_t offset = (uint32_t) vcpu->kvm_run->io.data_offset;
				uint32_t* data_address = (uint32_t*)(p+offset);
				uint32_t op =total_exits;
				// *data_address = total_exits;
				// printf("Address kvm %p \n",p + vcpu->kvm_run->io.data_offset);
				// printf("Address pointer op %p \n",&op);
				// vcpu->kvm_run->io.size = sizeof(uint32_t);
				memcpy( data_address, &op,sizeof(uint32_t));
				
				// printf("Val in ptr %d \n",*((uint32_t *)(p + vcpu->kvm_run->io.data_offset)));
				// printf("KVM blocked KVM_EXIT_IO F3 %d \n",vcpu->kvm_run->exit_reason);
				continue;
			}

			/* fall through */
		default:
			printf("KVM blocked default %d \n",vcpu->kvm_run->exit_reason);
			fprintf(stderr,	"Got exit_reason %d,"
				" expected KVM_EXIT_HLT (%d)\n",
				vcpu->kvm_run->exit_reason, KVM_EXIT_HLT);
			exit(1);
		}
	}

 check:
 	printf("KVM blocked check %d \n",vcpu->kvm_run->exit_reason);		
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
		.base = 0,
		.limit = 0xffffffff,
		.selector = 1 << 3,
		.present = 1,
		.type = 11, /* Code: execute, read, accessed */
		.dpl = 0,
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
	sregs->cr4 = CR4_PAE;
	sregs->cr0
		= CR0_PE | CR0_MP | CR0_ET | CR0_NE | CR0_WP | CR0_AM | CR0_PG;
	sregs->efer = EFER_LME | EFER_LMA;

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
	} mode = REAL_MODE;
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
