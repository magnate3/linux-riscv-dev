#include <fcntl.h>
#include <linux/kvm.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <errno.h>

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
struct vm *g_vm;

int setup_vm(struct vm *vm, int ram_size) {
    int ret = 0;

    // create vm
    if ((vm->vm_fd = ioctl(g_dev_fd, KVM_CREATE_VM, 0)) < 0) {
	fprintf(stderr, "failed to create vm.\n");
	return -1;
    }

    // create irpchip
    ret = ioctl(vm->vm_fd, KVM_CREATE_IRQCHIP);
    if (ret < 0) {
	fprintf(stderr, "failed to create irqchip. \n");
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

    // mmap kvm_run
    int run_size = ioctl(g_dev_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
    vcpu->run =
	mmap(NULL, run_size, PROT_READ | PROT_WRITE, MAP_SHARED, vcpu->fd, 0);
    if (vcpu->run == MAP_FAILED) {
	fprintf(stderr, "failed to map run for vm. \n");
	return -1;
    }
}

void setup_timer() {
    struct itimerspec its;
    struct sigevent sev;
    timer_t timerid;

    memset(&sev, 0, sizeof(sev));
    sev.sigev_value.sival_int = 0;
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = SIGALRM;

    if (timer_create(CLOCK_REALTIME, &sev, &timerid) < 0)
	fprintf(stderr, "failed to create timer. \n");

    memset(&its, 0, sizeof(its));

    its.it_value.tv_sec = 1;
    its.it_interval.tv_sec = 1;

    if (timer_settime(timerid, 0, &its, NULL) < 0)
	fprintf(stderr, "failed to set timer. \n");
}

void kvm_irq_line(int irq, int level) {
    struct kvm_irq_level irq_level;

    irq_level = (struct kvm_irq_level){{
					   .irq = irq,
				       },
				       .level = level};

    if (ioctl(g_vm->vm_fd, KVM_IRQ_LINE, &irq_level) < 0) {
	fprintf(stderr, "failed to set kvm irq line. \n");
    }
}

void serial_int(int sig) {
    kvm_irq_line(4, 0);
    kvm_irq_line(4, 1);
}

void load_image(struct vm *vm) {
    int ret = 0;
    int fd = open("./serial-io/guest/kernel.bin", O_RDONLY);
    if (fd < 0) {
	fprintf(stderr, "can not open guest image \n");
	exit(-1);
    }

    char *p = (char *)vm->ram_start + ((0x1000 << 4) + 0x0);

    while (1) {
	if ((ret = read(fd, p, 4096)) <= 0) break;

	p += ret;
    }
}

void run_vm(struct vm *vm) {
    int ret = 0;

    while (1) {
	ret = ioctl(vm->vcpu[0]->fd, KVM_RUN, 0);
	if (ret == -1 && errno == EINTR) {
	    continue;
	}
	if (ret < 0) {
	    continue;
	    fprintf(stderr, "failed to run kvm.\n");
	    exit(1);
	}

	switch (vm->vcpu[0]->run->exit_reason) {
	    case KVM_EXIT_IO
		: kvm_emulate_io(vm->vcpu[0]->run->io.port,
				 vm->vcpu[0]->run->io.direction,
				 (char *)vm->vcpu[0]->run +
				     vm->vcpu[0]->run->io.data_offset);
	    sleep(1);
	    break;
	defaut:
	    break;
	}
    }
}

void serial_in(char *data){
    static int c = 0;
    *data = c++;
}

void serial_out(void *data) {
    fprintf(stdout, "tx data: %d \n", *(char *)data);
}

void kvm_emulate_io(uint16_t port, uint8_t direction, void *data) {
    if (port == 0x3f8) {
	if (direction == KVM_EXIT_IO_OUT) {
	    serial_out(data);
	} else {
	    serial_in(data);
	}
    }
}

int main(int argc, char **argv) {
    if ((g_dev_fd = open("/dev/kvm", O_RDWR)) < 0) {
	fprintf(stderr, "failed to open KVM device. \n");
	return -1;
    }

    g_vm = malloc(sizeof(struct vm));
    struct vcpu *vcpu = malloc(sizeof(struct vcpu));
    vcpu->id = 0;
    g_vm->vcpu[0] = vcpu;

    setup_vm(g_vm, 64000000);
    load_image(g_vm);

    signal(SIGALRM, serial_int);
    setup_timer();

    run_vm(g_vm);

    return 0;
}
