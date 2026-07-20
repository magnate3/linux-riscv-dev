#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "pcidma.h"
#include <sys/ioctl.h>
int main(int argc, char **argv)
{
    int fd;
    struct args_enable args;

    args.pci_loc.domain = 0;
    args.pci_loc.bus = 0x5;
    args.pci_loc.slot = 0;
    args.pci_loc.func = 0;

    fd = open("/dev/pcidma", O_RDONLY);
    ioctl(fd, PCIDMA_ENABLE, &args);
    /* do_work(); */
    return 0;
}
