#include <fcntl.h>
#include <linux/kvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

struct vm {
    int vm_fd;
    __u64 ram_size;
    __u64 ram_start;
    struct kvm_userspace_memory_region mem;
    struct vcpu *vcpu[1];
};

struct vcpu {
    int id;
    int fd;
    struct kvm_run *run;
    struct kvm_sregs sregs;
    struct kvm_regs regs;
};

int g_dev_fd;

int setup_vm(struct vm *vm, int ram_size) {
    int ret = 0;

    if ((vm->vm_fd = ioctl(g_dev_fd, KVM_CREATE_VM, 0)) < 0) {
	fprintf(stderr, "failed to create vm.\n");
	return -1;
    }

    vm->ram_size = ram_size;
    vm->ram_start =
	(__u64)mmap(NULL, vm->ram_size, PROT_READ | PROT_WRITE,
		    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

    if ((void *)vm->ram_start == MAP_FAILED) {
	fprintf(stderr, "failed to map memory for vm. \n");
	return -1;
    }

    vm->mem.slot = 0;
    vm->mem.guest_phys_addr = 0;
    vm->mem.memory_size = vm->ram_size;
    vm->mem.userspace_addr = vm->ram_start;

    if ((ioctl(vm->vm_fd, KVM_SET_USER_MEMORY_REGION, &(vm->mem))) < 0) {
	fprintf(stderr, "failed to set memory for vm. \n");
	return -1;
    }

    struct vcpu *vcpu = vm->vcpu[0];
    vcpu->fd = ioctl(vm->vm_fd, KVM_CREATE_VCPU, vcpu->id);
    if (vm->vcpu[0]->fd < 0) {
	fprintf(stderr, "failed to create for vm. \n");
    }

    // sregs
    if (ioctl(vcpu->fd, KVM_GET_SREGS, &(vcpu->sregs)) < 0) {
	fprintf(stderr, "failed to get sregs.\n");
	exit(-1);
    }
    vcpu->sregs.cs.selector = 0x1000;
    vcpu->sregs.cs.base = 0x1000 << 4;
    if (ioctl(vcpu->fd, KVM_SET_SREGS, &(vcpu->sregs)) < 0) {
	fprintf(stderr, "failed to set sregs.\n");
	exit(-1);
    }

    // regs
    if (ioctl(vcpu->fd, KVM_GET_REGS, &(vcpu->regs)) < 0) {
	fprintf(stderr, "failed to get regs.\n");
	exit(-1);
    }
    vcpu->regs.rip = 0x0;
    vcpu->regs.rflags = 0x2;
    if (ioctl(vcpu->fd, KVM_SET_REGS, &(vcpu->regs)) < 0) {
	fprintf(stderr, "failed to set regs.\n");
	exit(-1);
    }
}

void load_image(struct vm *vm) {
    int ret = 0;
    int fd = open("./quick-start/guest/kernel.bin", O_RDONLY);
    if (fd < 0) {
	fprintf(stderr, "can not open guest image \n");
	exit(-1);
    }

    char *p = (char *)vm->ram_start + ((0x1000 << 4) + 0x0);

    while (1) {
	if ((ret = read(fd, p, 4096)) <= 0)
	    break;

	p += ret;
    }
}

void run_vm(struct vm *vm) {
    int ret = 0;

    while (1) {
	if (ioctl(vm->vcpu[0]->fd, KVM_RUN, 0) < 0) {
	    fprintf(stderr, "failed to run kvm.\n");
	    exit(1);
	}
    }
}

int main(int argc, char **argv) {
    if ((g_dev_fd = open("/dev/kvm", O_RDWR)) < 0) {
	fprintf(stderr, "failed to open KVM device. \n");
	return -1;
    }

    struct vm *vm = malloc(sizeof(struct vm));
    struct vcpu *vcpu = malloc(sizeof(struct vcpu));
    vcpu->id = 0;
    vm->vcpu[0] = vcpu;

    setup_vm(vm, 64000000);
    load_image(vm);
    run_vm(vm);

    return 0;
}
