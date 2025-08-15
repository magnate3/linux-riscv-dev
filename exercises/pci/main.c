#include <stdio.h>
#include <string.h>
#include "pci_common.h"

int main(int argc, char *argv[])
{
	pci_scan();

	pci_uio_alloc_resource();
}
