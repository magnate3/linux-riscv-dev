#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

/* map a particular resource from a file */
void *
pci_map_resource(void *requested_addr, int fd, off_t offset, size_t size,
                 int additional_flags)
{
        void *mapaddr;

        /* Map the PCI memory resource of device */
        mapaddr = mmap(requested_addr, size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | additional_flags, fd, offset);
        if (mapaddr == MAP_FAILED) {
                RTE_LOG(ERR, EAL,

                        __func__, fd, requested_addr, size,
                        (unsigned long long)offset,
                        strerror(errno), mapaddr);
        } else
                RTE_LOG(DEBUG, EAL, "  PCI memory mapped at %p\n", mapaddr);

        return mapaddr;
}
int main()
{
	int uiofd;
	int configfd;
	int err;
	int i;
	unsigned icount;
	unsigned char command_high;

	uiofd = open("/dev/uio0", O_RDONLY);
	if (uiofd < 0) {
		perror("uio open:");
		return errno;
	}
	configfd = open("/sys/class/uio/uio0/device/config", O_RDWR);
	if (uiofd < 0) {
		perror("config open:");
		return errno;
	}

	/* Read and cache command value */
	err = pread(configfd, &command_high, 1, 5);
	if (err != 1) {
		perror("command config read:");
		return errno;
	}
	command_high &= ~0x4;

	for(i = 0;; ++i) {
		/* Print out a message, for debugging. */
		if (i == 0)
			fprintf(stderr, "Started uio test driver.\n");
		else
			fprintf(stderr, "Interrupts: %d\n", icount);

		/****************************************/
		/* Here we got an interrupt from the
		   device. Do something to it. */
		/****************************************/

		/* Re-enable interrupts. */
		err = pwrite(configfd, &command_high, 1, 5);
		if (err != 1) {
			perror("config write:");
			break;
		}

		/* Wait for next interrupt. */
		err = read(uiofd, &icount, 4);
		if (err != 4) {
			perror("uio read:");
			break;
		}

	}
	return errno;
}
