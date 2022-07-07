#include "pci_common.h"
#include <string.h>
//#include <inttypes.h>
#include <stdint.h>
//#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
//#include <sys/queue.h>
#include <sys/mman.h>

#include "pci.h"

#include "common_string_fns.h"


#define HBA_VERDOR_ID     0x10ee
#define HBA_DEVICE_ID     0x7014

#define SYSFS_PCI_DEVICES "/sys/bus/pci/devices"

char pci_dev_path[PATH_MAX];

struct pci_addr hba_pci_addr;

const char *pci_get_sysfs_path(void)
{
	return SYSFS_PCI_DEVICES;
}

/*
 * split up a pci address into its constituent parts.
 */
static int parse_pci_addr_format(const char *buf, int bufsize, struct pci_addr *addr)
{
	/* first split on ':' */
	union splitaddr {
		struct {
			char *domain;
			char *bus;
			char *devid;
			char *function;
		};
		char *str[PCI_FMT_NVAL]; /* last element-separator is "." not ":" */
	} splitaddr;

	char *buf_copy = strndup(buf, bufsize);
	if (buf_copy == NULL)
		return -1;

	if (strsplit(buf_copy, bufsize, splitaddr.str, PCI_FMT_NVAL, ':') != PCI_FMT_NVAL - 1)
		goto error;
	/* final split is on '.' between devid and function */
	splitaddr.function = strchr(splitaddr.devid,'.');
	if (splitaddr.function == NULL)
		goto error;
	*splitaddr.function++ = '\0';

	/* now convert to int values */
	errno = 0;
	addr->domain = strtoul(splitaddr.domain, NULL, 16);
	addr->bus = strtoul(splitaddr.bus, NULL, 16);
	addr->devid = strtoul(splitaddr.devid, NULL, 16);
	addr->function = strtoul(splitaddr.function, NULL, 10);
	if (errno != 0)
		goto error;

	free(buf_copy); /* free the copy made with strdup */
	return 0;
error:
	free(buf_copy);
	return -1;
}

/* parse a sysfs (or other) file containing one integer value */
int parse_sysfs_value(const char *filename, unsigned long *val)
{
	FILE *f;
	char buf[BUFSIZ];
	char *end = NULL;

	if ((f = fopen(filename, "r")) == NULL) {
		printf("%s(): cannot open sysfs value %s\n", __func__, filename);
		return -1;
	}

	if (fgets(buf, sizeof(buf), f) == NULL) {
		printf("%s(): cannot read sysfs value %s\n",__func__, filename);
		fclose(f);
		return -1;
	}
	*val = strtoul(buf, &end, 0);
	if ((buf[0] == '\0') || (end == NULL) || (*end != '\n')) {
		printf("%s(): cannot parse sysfs value %s\n", __func__, filename);
		fclose(f);
		return -1;
	}
	fclose(f);
	return 0;
}


/* Scan one pci sysfs entry, and fill the devices list from it. */
static int pci_scan_one(const char *dirname, const struct pci_addr *addr)
{
	char filename[PATH_MAX];
	unsigned long tmp;
	struct pci_id id;
	int ret;

	/* get vendor id */
	snprintf(filename, sizeof(filename), "%s/vendor", dirname);
	if (parse_sysfs_value(filename, &tmp) < 0) {
		return -1;
	}
	id.vendor_id = (uint16_t)tmp;

	/* get device id */
	snprintf(filename, sizeof(filename), "%s/device", dirname);
	if (parse_sysfs_value(filename, &tmp) < 0) {
		return -1;
	}
	id.device_id = (uint16_t)tmp;

	/* get subsystem_vendor id */
	snprintf(filename, sizeof(filename), "%s/subsystem_vendor",
		 dirname);
	if (parse_sysfs_value(filename, &tmp) < 0) {
		return -1;
	}
	id.subsystem_vendor_id = (uint16_t)tmp;

	/* get subsystem_device id */
	snprintf(filename, sizeof(filename), "%s/subsystem_device",
		 dirname);
	if (parse_sysfs_value(filename, &tmp) < 0) {
		return -1;
	}
	id.subsystem_device_id = (uint16_t)tmp;

	/* get class_id */
	snprintf(filename, sizeof(filename), "%s/class",
		 dirname);
	if (parse_sysfs_value(filename, &tmp) < 0) {
		return -1;
	}
	/* the least 24 bits are valid: class, subclass, program interface */
	id.class_id = (uint32_t)tmp & CLASS_ANY_ID;

	/*
	printf("class:0x%x vender:%#x, device:%#x, sub vendor:%#x, sub devid:%#x\n",
		id.class_id, id.vendor_id, id.device_id, id.subsystem_vendor_id, id.subsystem_device_id);
	*/
	if (id.vendor_id == HBA_VERDOR_ID && id.device_id == HBA_DEVICE_ID) {
		return 0;
	} else {
		return -1;
	}
#if 0
	/* get max_vfs */
	dev->max_vfs = 0;
	snprintf(filename, sizeof(filename), "%s/max_vfs", dirname);
	if (!access(filename, F_OK) &&
	    eal_parse_sysfs_value(filename, &tmp) == 0)
		dev->max_vfs = (uint16_t)tmp;
	else {
		/* for non igb_uio driver, need kernel version >= 3.8 */
		snprintf(filename, sizeof(filename),
			 "%s/sriov_numvfs", dirname);
		if (!access(filename, F_OK) &&
		    eal_parse_sysfs_value(filename, &tmp) == 0)
			dev->max_vfs = (uint16_t)tmp;
	}

	/* get numa node, default to 0 if not present */
	snprintf(filename, sizeof(filename), "%s/numa_node",
		 dirname);

	if (access(filename, F_OK) != -1) {
		if (eal_parse_sysfs_value(filename, &tmp) == 0)
			dev->device.numa_node = tmp;
		else
			dev->device.numa_node = -1;
	} else {
		dev->device.numa_node = 0;
	}

	pci_name_set(dev);

	/* parse resources */
	snprintf(filename, sizeof(filename), "%s/resource", dirname);
	if (pci_parse_sysfs_resource(filename, dev) < 0) {
		RTE_LOG(ERR, EAL, "%s(): cannot parse resource\n", __func__);
		free(dev);
		return -1;
	}

	/* parse driver */
	snprintf(filename, sizeof(filename), "%s/driver", dirname);
	ret = pci_get_kernel_driver_by_path(filename, driver, sizeof(driver));
	if (ret < 0) {
		RTE_LOG(ERR, EAL, "Fail to get kernel driver\n");
		free(dev);
		return -1;
	}

	if (!ret) {
		if (!strcmp(driver, "vfio-pci"))
			dev->kdrv = RTE_KDRV_VFIO;
		else if (!strcmp(driver, "igb_uio"))
			dev->kdrv = RTE_KDRV_IGB_UIO;
		else if (!strcmp(driver, "uio_pci_generic"))
			dev->kdrv = RTE_KDRV_UIO_GENERIC;
		else
			dev->kdrv = RTE_KDRV_UNKNOWN;
	} else
		dev->kdrv = RTE_KDRV_NONE;

	/* device is valid, add in list (sorted) */
	if (TAILQ_EMPTY(&rte_pci_bus.device_list)) {
		rte_pci_add_device(dev);
	} else {
		struct rte_pci_device *dev2;
		int ret;

		TAILQ_FOREACH(dev2, &rte_pci_bus.device_list, next) {
			ret = rte_pci_addr_cmp(&dev->addr, &dev2->addr);
			if (ret > 0)
				continue;

			if (ret < 0) {
				rte_pci_insert_device(dev2, dev);
			} else { /* already registered */
				if (!rte_dev_is_probed(&dev2->device)) {
					dev2->kdrv = dev->kdrv;
					dev2->max_vfs = dev->max_vfs;
					pci_name_set(dev2);
					memmove(dev2->mem_resource,
						dev->mem_resource,
						sizeof(dev->mem_resource));
				} else {
					/**
					 * If device is plugged and driver is
					 * probed already, (This happens when
					 * we call rte_dev_probe which will
					 * scan all device on the bus) we don't
					 * need to do anything here unless...
					 **/
					if (dev2->kdrv != dev->kdrv ||
						dev2->max_vfs != dev->max_vfs)
						/*
						 * This should not happens.
						 * But it is still possible if
						 * we unbind a device from
						 * vfio or uio before hotplug
						 * remove and rebind it with
						 * a different configure.
						 * So we just print out the
						 * error as an alarm.
						 */
						RTE_LOG(ERR, EAL, "Unexpected device scan at %s!\n",
							filename);
				}
				free(dev);
			}
			return 0;
		}

		rte_pci_add_device(dev);
	}
#endif
	return 0;
}

int pci_scan(void)
{
	struct dirent *e;
	DIR *dir;
	char dirname[PATH_MAX];
	struct pci_addr addr;

	dir = opendir(pci_get_sysfs_path());
	if (dir == NULL) {
		printf("%s(): opendir failed: %s\n", __func__, strerror(errno));
		return -1;
	}

	while ((e = readdir(dir)) != NULL) {
		if (e->d_name[0] == '.')
			continue;

		if (parse_pci_addr_format(e->d_name, sizeof(e->d_name), &addr) != 0)
			continue;

		snprintf(dirname, sizeof(dirname), "%s/%s", pci_get_sysfs_path(), e->d_name);

		//printf("%s/%s: domain:%u, bus:%u devid:%u func:%u\n", pci_get_sysfs_path(), e->d_name, addr.domain, addr.bus, addr.devid, addr.function);
		if (pci_scan_one(dirname, &addr) == 0) {
			memcpy(&hba_pci_addr, &addr, sizeof(addr));
			printf("domain:%u, bus:%u, dev:%u, func:%u\n", 
				hba_pci_addr.domain, hba_pci_addr.bus, hba_pci_addr.devid, hba_pci_addr.function);
			break;
		}
	}

	closedir(dir);
	return 0;
}



int pci_uio_alloc_resource(void)
{
	char devname[PATH_MAX];
	int uio_num;
	int fd;
	int fd_resource;
	void *mapaddr;
	uint32_t val = 0;
	uint8_t *testaddr;

	uio_num = 0;
	snprintf(devname, sizeof(devname), "/dev/uio%u", uio_num);
	fd = open(devname, O_RDWR);
	if (fd < 0) {
		printf("cannot open:%s: %s\n", devname, strerror(errno));
		return -1;
	}

	snprintf(devname, sizeof(devname), "%s/" PCI_PRI_FMT "/resource0",
		 pci_get_sysfs_path(),
		 hba_pci_addr.domain, hba_pci_addr.bus, hba_pci_addr.devid, hba_pci_addr.function);
	/* then try to map resource file */
	fd_resource = open(devname, O_RDWR);
	if (fd_resource < 0) {
		printf("Cannot open %s: %s\n", devname, strerror(errno));

		close(fd);
		return -2;
	}

	mapaddr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (mapaddr == MAP_FAILED) {
		printf("mmap failed\n");
	}

	val = *(uint32_t *)mapaddr;
	printf("val = 0x%x\n", val);

	testaddr = (uint8_t *)mapaddr;
	val = *(uint32_t *)(testaddr + 0x4);
	printf("val = 0x%x\n", val);

	munmap(mapaddr, 0x1000);

	close(fd_resource);
	close(fd);

	return 0;
}

