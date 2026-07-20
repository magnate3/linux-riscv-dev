#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/limits.h>

/** IO resource type: */
#define IORESOURCE_IO         0x00000100
#define IORESOURCE_MEM        0x00000200

#define PCI_MAX_RESOURCE 6
#define PCI_RESOURCE_FMT_NVAL 3

#define PCI_PRI_FMT "%.4" PRIx32 ":%.2" PRIx8 ":%.2" PRIx8 ".%" PRIx8
#define PCI_SHORT_PRI_FMT "%.2" PRIx8 ":%.2" PRIx8 ".%" PRIx8

struct mem_resource {
        uint64_t phys_addr; /**< Physical address, 0 if not resource. */
        uint64_t len;       /**< Length of the resource. */
        void *addr;         /**< Virtual address, NULL when not mapped. */
};

struct pci_addr {
        int domain;
        int bus;
        int devid;
        int function;
};

struct pci_device {
        struct pci_addr loc;
        struct mem_resource mem_resource[PCI_MAX_RESOURCE];
};

/* split string into tokens */
int strsplit(char *string, int stringlen,
             char **tokens, int maxtokens, char delim)
{
        int i, tok = 0;
        int tokstart = 1; /* first token is right at start of string */

        if (string == NULL || tokens == NULL)
                goto einval_error;

        for (i = 0; i < stringlen; i++) {
                if (string[i] == '\0' || tok >= maxtokens)
                        break;
                if (tokstart) {
                        tokstart = 0;
                        tokens[tok++] = &string[i];
                }
                if (string[i] == delim) {
                        string[i] = '\0';
                        tokstart = 1;
                }
        }
        return tok;

einval_error:
        errno = EINVAL;
        return -1;
}
/* split string into tokens */
int
rte_strsplit(char *string, int stringlen,
             char **tokens, int maxtokens, char delim)
{
        int i, tok = 0;
        int tokstart = 1; /* first token is right at start of string */

        if (string == NULL || tokens == NULL)
                goto einval_error;

        for (i = 0; i < stringlen; i++) {
                if (string[i] == '\0' || tok >= maxtokens)
                        break;
                if (tokstart) {
                        tokstart = 0;
                        tokens[tok++] = &string[i];
                }
                if (string[i] == delim) {
                        string[i] = '\0';
                        tokstart = 1;
                }
        }
        return tok;

einval_error:
        errno = EINVAL;
        return -1;
}
/* parse one line of the "resource" sysfs file (note that the 'line'
 * string is modified)
 */
int pci_parse_one_sysfs_resource(char *line, size_t len, uint64_t *phys_addr,
                                 uint64_t *end_addr, uint64_t *flags)
{
        union pci_resource_info {
                struct {
                        char *phys_addr;
                        char *end_addr;
                        char *flags;
                };
                char *ptrs[PCI_RESOURCE_FMT_NVAL];
        } res_info;
#if 0
        if (strsplit(line, len, res_info.ptrs, 3, ' ') != 3) {
                fprintf(stderr,
                        "%s(): bad resource format\n", __func__);
                return -1;
        }
#else
        if (rte_strsplit(line, len, res_info.ptrs, 3, ' ') != 3) {
                fprintf(stderr,
                        "%s(): bad resource format\n", __func__);
                return -1;
        }
#endif
        errno = 0;
        *phys_addr = strtoull(res_info.phys_addr, NULL, 16);
        *end_addr = strtoull(res_info.end_addr, NULL, 16);
        *flags = strtoull(res_info.flags, NULL, 16);
        //printf("%s res_info.phys_addr %s and phys_addr %016lx\n", __func__, res_info.phys_addr, *phys_addr);
        if (errno != 0) {
                fprintf(stderr,
                        "%s(): bad resource format\n", __func__);
                return -1;
        }

        return 0;
}


/* parse the "resource" sysfs file */
static int pci_parse_sysfs_resource(struct pci_device *dev)
{
        FILE *f;
        char buf[BUFSIZ];
        int i;
        uint64_t phys_addr, end_addr, flags;
        char path[PATH_MAX];

        sprintf(path, "/sys/bus/pci/devices/0000:" PCI_SHORT_PRI_FMT "/resource",
                 dev->loc.bus, dev->loc.devid, dev->loc.function);

        printf("%s path is %s \n",__func__, path);
        f = fopen(path, "r");
        if (f == NULL) {
                fprintf(stderr, "Cannot open sysfs resource\n");
                return -1;
        }

        for (i = 0; i < PCI_MAX_RESOURCE; i++) {

                if (fgets(buf, sizeof(buf), f) == NULL) {
                        fprintf(stderr,
                                "%s(): cannot read resource\n", __func__);
                        goto error;
                }
                if (pci_parse_one_sysfs_resource(buf, sizeof(buf), &phys_addr,
                                &end_addr, &flags) < 0)
                        goto error;

                if (flags & IORESOURCE_MEM) {
                        dev->mem_resource[i].phys_addr = phys_addr;
                        dev->mem_resource[i].len = end_addr - phys_addr + 1;
                        /* not mapped for now */
                        dev->mem_resource[i].addr = NULL;

                        printf("Resource #%d:\n", i);
                        printf("\tphys_addr = 0x%016lx:\n", dev->mem_resource[i].phys_addr);
                        printf("\tlen       = 0x%016lx:\n", dev->mem_resource[i].len);
                }
        }
        fclose(f);
        return 0;

error:
        fclose(f);
        return -1;
}

static int pci_resource_mmap(struct pci_device *dev, int index)
{
        int fd;
        char path[PATH_MAX];
        char *addr;

        if (dev->mem_resource[index].phys_addr == 0) {
                fprintf(stderr,
                        "No resource defined at BAR#%d\n", index);
                return -EINVAL;
        }

        sprintf(path, "/sys/bus/pci/devices/0000:" PCI_SHORT_PRI_FMT "/resource%d",
                 dev->loc.bus, dev->loc.devid, dev->loc.function, index);
        printf("%s path is %s \n",__func__, path);
        fd = open(path, O_RDWR);
        if (fd < 0) {
                fprintf(stderr, "Failed to open %s\n", path);
                return -errno;
        };

        addr = mmap(NULL, (size_t)dev->mem_resource[index].len,
                    PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == (void *) -1) {
                fprintf(stderr,
                        "Failed to mmap: (%d) [%s]\n", errno, strerror(errno));
                return -errno;
        }

        dev->mem_resource[index].addr = addr;
        return 0;
}

void print_usage(int argc, char *argv[])
{
        //                      argv[0] [1]     [2]     [3]      [4]       [5]      [6]
        fprintf(stderr, "\nUsage:\t%s { op } { bfd } { bar } { offset } { width } [ data ]\n"
                "\top      : operation type: [r]ead, [w]rite \n"
                "\tbdf     : bdf of device to act on, e.g. 02:00.0 \n"
                "\tbar     : pci bar/memory region to act on\n"
                "\toffset  : offset into pci memory region\n"
                "\twidth   : number of bytes\n"
                "\tdata    : data to be written\n\n",
                argv[0]);
}

#define rte_bswap32(x) __builtin_bswap32(x)
#define rte_be_to_cpu_32(x) rte_bswap32(x)
#define rte_wmb() do { asm volatile ("dmb st" : : : "memory"); } while (0)
#define rte_rmb() __sync_synchronize()
#define rte_io_rmb() rte_rmb()
uint32_t
rte_read32_relaxed(const volatile void *addr)
{
        return *(const volatile uint32_t *)addr;
}
uint32_t
rte_read32(const volatile void *addr)
{
        uint32_t val;
        val = rte_read32_relaxed(addr);
        rte_io_rmb();
        return val;
}
static inline uint32_t hinic_hwif_read_reg(void *  reg)
{
        return  rte_be_to_cpu_32(rte_read32(reg));
}
int main(int argc, char *argv[]) {
        struct pci_device dev = { 0 };
        int op = 'w';
        int bar, width, offset, data;
        uint8_t *addr;
        int ret;

        /* Parse options */
        if(argc < 6) {
                print_usage(argc, argv);
                exit(1);
        }

        if (!strcmp(argv[1], "r") || !strcmp(argv[1], "read"))
                op = 'r';
        else if (!strcmp(argv[1], "w") || !strcmp(argv[1], "write")) {
                op = 'w';
                if (argc < 7) {
                        print_usage(argc, argv);
                        exit(1);
                }
        } else {
                fprintf(stderr, "Illegal op type '%c'\n", argv[1]);
                exit(1);
        }

        ret = sscanf(argv[2], "%x:%x.%x",
                     &dev.loc.bus, &dev.loc.devid, &dev.loc.function);
        if (ret != 3) {
                fprintf(stderr, "Illegal pci address: %s\n", argv[2]);
                exit(1);
        }

        bar = strtol(argv[3], NULL, 0);
        offset = strtol(argv[4], NULL, 0);
        width = strtol(argv[5], NULL, 0);
        if (op == 'w') {
                data = strtol(argv[6], NULL, 0);
        }
        /* mmap */
        ret = pci_parse_sysfs_resource(&dev);
        if (ret)
                exit(1);

        ret = pci_resource_mmap(&dev, bar);
        if (ret)
                exit(1);

        printf("\n");

        /* read/write */
        addr = (uint8_t *)dev.mem_resource[bar].addr + offset;

        if (op == 'w') {
                switch (width) {
                case 1:
                        *((uint8_t*)addr) = data;
                        break;
                case 2:
                        *((uint16_t*)addr) = data;
                        break;
                case 4:
                        *((uint32_t*)addr) = data;
                        break;
                case 8:
                        *((uint64_t*)addr) = data;
                        break;
                default:
                        fprintf(stderr, "Illegal width\n");
                        exit(1);
                }
        }

        switch (width) {
        case 1:
                printf("@0x%08x = 0x%02x\n", offset, *((uint8_t*)addr));
                break;
        case 2:
                printf("@0x%08x = 0x%04x\n", offset, *((uint16_t*)addr));
                break;
        case 4:
                //printf("@0x%08x = 0x%08x\n", offset, *((uint32_t*)addr));
                printf("@0x%08x = 0x%08x\n", offset, hinic_hwif_read_reg(addr));
                break;
        case 8:
                printf("@0x%08x = 0x%016x\n", offset, *((uint64_t*)addr));
                break;
        default:
                fprintf(stderr, "Illegal width\n");
                exit(1);
        }

        return 0;
}

