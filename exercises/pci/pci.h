#ifndef _PCI_H_
#define _PCI_H_

/**
 * @file
 *
 * Common PCI Library
 */

#include <limits.h>
#include <stdint.h>
#include <inttypes.h>

/* Formatting string for PCI device identifier: Ex: 0000:00:01.0 */
#define PCI_PRI_FMT "%.4" PRIx16 ":%.2" PRIx8 ":%.2" PRIx8 ".%" PRIx8
#define PCI_PRI_STR_SIZE sizeof("XXXXXXXX:XX:XX.X")

/* Short formatting string, without domain, for PCI device: Ex: 00:01.0 */
#define PCI_SHORT_PRI_FMT "%.2" PRIx8 ":%.2" PRIx8 ".%" PRIx8

/* Nb. of values in PCI device identifier format string. */
#define PCI_FMT_NVAL 4

/* Nb. of values in PCI resource format. */
#define PCI_RESOURCE_FMT_NVAL 3

/* Maximum number of PCI resources. */
#define PCI_MAX_RESOURCE 6

/**
 * A structure describing an ID for a PCI driver. Each driver provides a
 * table of these IDs for each device that it supports.
 */
struct pci_id {
	uint32_t class_id;               /* Class ID or CLASS_ANY_ID. */
	uint16_t vendor_id;              /* Vendor ID or PCI_ANY_ID. */
	uint16_t device_id;              /* Device ID or PCI_ANY_ID. */
	uint16_t subsystem_vendor_id;    /* Subsystem vendor ID  or PCI_ANY_ID. */
	uint16_t subsystem_device_id;    /* Subsystem device ID or PCI_ANY_ID. */
};

/**
 * A structure describing the location of a PCI device.
 */
struct pci_addr {
	uint32_t domain;
	uint8_t bus;
	uint8_t devid;
	uint8_t function;
};

/** Any PCI device identifier (vendor, device, ...) */
#define PCI_ANY_ID (0xffff)
#define CLASS_ANY_ID (0xffffff)


/**
 * A structure describing a PCI mapping.
 */
struct pci_map {
	void *addr;
	char *path;
	uint64_t offset;
	uint64_t size;
	uint64_t phaddr;
};

#endif
