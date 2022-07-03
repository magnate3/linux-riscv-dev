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
#define UIO_DEV "/dev/uio0"
#define UIO_ADDR "/sys/class/uio/uio0/maps/map0/addr"
#define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"
/* HW interface registers */
#define HINIC_CSR_FUNC_ATTR0_ADDR                       0x0
#define HINIC_CSR_FUNC_ATTR1_ADDR                       0x4
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
static char uio_addr_buf[64]={0};
static char uio_size_buf[64]={0};
int main(int argc, char *argv[]) {
    int uio_fd,addr_fd,size_fd;
    uint64_t uio_size;
    uint64_t *uio_addr;
    void *access_address;
    void *addr0, *addr1;
    int n=0;
    uio_fd = open(UIO_DEV,O_RDWR);
    addr_fd = open(UIO_ADDR,O_RDONLY);
    size_fd = open(UIO_SIZE,O_RDONLY);
    u_int32_t status = 0;
    if(addr_fd < 0 || size_fd < 0 || uio_fd < 0){
        fprintf(stderr,"mmap:%d\n",errno);
        exit(-1);
    }

    n=read(addr_fd,uio_addr_buf,sizeof(uio_addr_buf));
    if(n<0){
        fprintf(stderr, "%d\n", errno);
        exit(-1);
    }
    n=read(size_fd,uio_size_buf,sizeof(uio_size_buf));
    if(n<0){
        fprintf(stderr, "%d\n", errno);
        exit(-1);
    }
    uio_addr = (uint64_t*)strtoull(uio_addr_buf,NULL,16);
    uio_size = (uint64_t)strtoull(uio_size_buf,NULL,16);

    access_address = mmap(NULL,uio_size,PROT_READ | PROT_WRITE,
                            MAP_SHARED,uio_fd,0);
    if(access_address == (void*)-1){
        fprintf(stderr,"mmap:%d\n",errno);
        exit(-1);
    }
    addr0 = access_address + HINIC_CSR_FUNC_ATTR0_ADDR;
    addr1 = access_address + HINIC_CSR_FUNC_ATTR1_ADDR;
    printf("The device address %p (lenth %016xl)\n"
        "can be accessed over\n"
        "logical address %p\n",uio_addr,uio_size,access_address);
        printf("@0x%08x = 0x%08x\n",HINIC_CSR_FUNC_ATTR0_ADDR,  hinic_hwif_read_reg(addr0));
        printf("@0x%08x = 0x%08x\n",HINIC_CSR_FUNC_ATTR1_ADDR,  hinic_hwif_read_reg(addr1));
        return 0;
}

