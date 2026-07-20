#define _GNU_SOURCE
#include <asm/bootparam.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/kvm.h>
#include <linux/kvm_para.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct guest {
  int kvm_fd;
  int vm_fd;
  int vcpu_fd;
  void *mem;
};

static int guest_error(struct guest *g, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, ", errno: %d\n", errno);
  va_end(args);
  return -1;
}

static int guest_init_regs(struct guest *g) {
  struct kvm_regs regs;
  struct kvm_sregs sregs;
  if (ioctl(g->vcpu_fd, KVM_GET_SREGS, &(sregs)) < 0) {
    return guest_error(g, "failed to get registers");
  }

  sregs.cs.base = 0;
  sregs.cs.limit = ~0;
  sregs.cs.g = 1;

  sregs.ds.base = 0;
  sregs.ds.limit = ~0;
  sregs.ds.g = 1;

  sregs.fs.base = 0;
  sregs.fs.limit = ~0;
  sregs.fs.g = 1;

  sregs.gs.base = 0;
  sregs.gs.limit = ~0;
  sregs.gs.g = 1;

  sregs.es.base = 0;
  sregs.es.limit = ~0;
  sregs.es.g = 1;

  sregs.ss.base = 0;
  sregs.ss.limit = ~0;
  sregs.ss.g = 1;

  sregs.cs.db = 1;
  sregs.ss.db = 1;
  sregs.cr0 |= 1; /* enable protected mode */

  if (ioctl(g->vcpu_fd, KVM_SET_SREGS, &sregs) < 0) {
    return guest_error(g, "failed to set special registers");
  }

  if (ioctl(g->vcpu_fd, KVM_GET_REGS, &(regs)) < 0) {
    return guest_error(g, "failed to get registers");
  }

  regs.rflags = 2;
  regs.rip = 0x100000;
  regs.rsi = 0x10000;

  if (ioctl(g->vcpu_fd, KVM_SET_REGS, &(regs)) < 0) {
    return guest_error(g, "failed to set registers");
  }
  return 0;
}

static int guest_init_cpu_id(struct guest *g) {
  struct {
    uint32_t nent;
    uint32_t padding;
    struct kvm_cpuid_entry2 entries[100];
  } kvm_cpuid;
  kvm_cpuid.nent = sizeof(kvm_cpuid.entries) / sizeof(kvm_cpuid.entries[0]);
  ioctl(g->kvm_fd, KVM_GET_SUPPORTED_CPUID, &kvm_cpuid);

  for (unsigned int i = 0; i < kvm_cpuid.nent; i++) {
    struct kvm_cpuid_entry2 *entry = &kvm_cpuid.entries[i];
    if (entry->function == KVM_CPUID_SIGNATURE) {
      entry->eax = KVM_CPUID_FEATURES;
      entry->ebx = 0x4b4d564b; // KVMK
      entry->ecx = 0x564b4d56; // VMKV
      entry->edx = 0x4d;       // M
    }
  }
  ioctl(g->vcpu_fd, KVM_SET_CPUID2, &kvm_cpuid);
  return 0;
}


int guest_init(struct guest *g) {
  if ((g->kvm_fd = open("/dev/kvm", O_RDWR)) < 0) {
    return guest_error(g, "failed to open /dev/kvm");
  }

  if ((g->vm_fd = ioctl(g->kvm_fd, KVM_CREATE_VM, 0)) < 0) {
    return guest_error(g, "failed to create vm");
  }

  if (ioctl(g->vm_fd, KVM_SET_TSS_ADDR, 0xffffd000) < 0) {
    return guest_error(g, "failed to set tss addr");
  }

  __u64 map_addr = 0xffffc000;
  if (ioctl(g->vm_fd, KVM_SET_IDENTITY_MAP_ADDR, &map_addr) < 0) {
    return guest_error(g, "failed to set identity map addr");
  }

  if (ioctl(g->vm_fd, KVM_CREATE_IRQCHIP, 0) < 0) {
    return guest_error(g, "failed to create irq chip");
  }

  struct kvm_pit_config pit = {
      .flags = 0,
  };
  if (ioctl(g->vm_fd, KVM_CREATE_PIT2, &pit) < 0) {
    return guest_error(g, "failed to create i8254 interval timer");
  }

  g->mem = mmap(NULL, 1 << 30, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (g->mem == NULL) {
    return guest_error(g, "failed to mmap vm memory");
  }

  struct kvm_userspace_memory_region region = {
      .slot = 0,
      .flags = 0,
      .guest_phys_addr = 0,
      .memory_size = 1 << 30,
      .userspace_addr = (__u64)g->mem,
  };
  if (ioctl(g->vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
    return guest_error(g, "failed to set user memory region");
  }

  if ((g->vcpu_fd = ioctl(g->vm_fd, KVM_CREATE_VCPU, 0)) < 0) {
    return guest_error(g, "failed to create vcpu");
  }

  guest_init_regs(g);
  guest_init_cpu_id(g);

  return 0;
}

int guest_load(struct guest *g, const char *image_path) {
  size_t datasz;
  void *data;
  int fd = open(image_path, O_RDONLY);
  if (fd < 0) {
    return 1;
  }
  struct stat st;
  fstat(fd, &st);
  data = mmap(0, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  datasz = st.st_size;
  close(fd);

  struct boot_params *boot =
      (struct boot_params *)(((uint8_t *)g->mem) + 0x10000);
  void *cmdline = (void *)(((uint8_t *)g->mem) + 0x20000);
  void *kernel = (void *)(((uint8_t *)g->mem) + 0x100000);

  memset(boot, 0, sizeof(struct boot_params));
  memmove(boot, data, sizeof(struct boot_params));
  size_t setup_sectors = boot->hdr.setup_sects;
  size_t setupsz = (setup_sectors + 1) * 512;
  boot->hdr.vid_mode = 0xFFFF; // VGA
  boot->hdr.type_of_loader = 0xFF;
  boot->hdr.ramdisk_image = 0x0;
  boot->hdr.ramdisk_size = 0x0;
  boot->hdr.loadflags |= CAN_USE_HEAP | 0x01 | KEEP_SEGMENTS;
  boot->hdr.heap_end_ptr = 0xFE00;
  boot->hdr.ext_loader_ver = 0x0;
  boot->hdr.cmd_line_ptr = 0x20000;
  memset(cmdline, 0, boot->hdr.cmdline_size);
  memcpy(cmdline, "console=ttyS0", 14);
  memmove(kernel, (char *)data + setupsz, datasz - setupsz);
  return 0;
}

void guest_exit(struct guest *g) {
  close(g->kvm_fd);
  close(g->vm_fd);
  close(g->vcpu_fd);
  munmap(g->mem, 1 << 30);
}

int guest_run(struct guest *g) {
  int run_size = ioctl(g->kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
  struct kvm_run *run =
      mmap(0, run_size, PROT_READ | PROT_WRITE, MAP_SHARED, g->vcpu_fd, 0);
  for (;;) {
    int ret = ioctl(g->vcpu_fd, KVM_RUN, 0);
    if (ret < 0) {
      return guest_error(g, "kvm_run failed");
    }

    switch (run->exit_reason) {
    case KVM_EXIT_IO:
      if (run->io.port == 0x3f8 && run->io.direction == KVM_EXIT_IO_OUT) {
        uint32_t size = run->io.size;
        uint64_t offset = run->io.data_offset;
        printf("%.*s", size * run->io.count, (char *)run + offset);
      } else if (run->io.port == 0x3f8 + 5 &&
                 run->io.direction == KVM_EXIT_IO_IN) {
        char *value = (char *)run + run->io.data_offset;
        *value = 0x20;
      }
      break;
    case KVM_EXIT_SHUTDOWN:
      printf("shutdown\n");
      return 0;
    default:
      printf("reason: %d\n", run->exit_reason);
      return -1;
    }
  }
}

int main(int argc, char *argv[]) {
  int r;
  struct guest vm;

  r = guest_init(&vm);
  if (r < 0) {
    fprintf(stderr, "failed to initialize guest vm: %d, errno=%d\n", r, errno);
    return 1;
  }

  r = guest_load(&vm, argv[1]);
  if (r < 0) {
    fprintf(stderr, "failed to load guest image: %d, errno=%d\n", r, errno);
    return 1;
  }

  guest_run(&vm);
  guest_exit(&vm);
  return 0;
}