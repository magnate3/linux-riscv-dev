/*
List Infiniband devices on the host.

Author: Jörn Schumacher <joern.schumacher@cern.ch>
*/

#include <stdio.h>
#include <stdlib.h>
#include "infiniband/verbs.h"

int
main(int argc, char** argv)
{
	struct ibv_device** devices;
	int num_devices;
	int i;

	devices = ibv_get_device_list(&num_devices);

	for(i=0; i<num_devices; i++)
	{
		uint64_t guid = ibv_get_device_guid(devices[i]);
		printf("%s \t%s \t%s \t%s \t0x%lx\n", 
		       devices[i]->name,
		       devices[i]->dev_name,
		       devices[i]->dev_path,
		       devices[i]->ibdev_path,
		       guid);
	}

	return EXIT_SUCCESS;
}
