/*
 * Copyright (C) 2017 Hewlett Packard Enterprise Development, L.P.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 */

struct ptdump_req {
	unsigned long addr;
	int order;
};

#define PTDUMP_BASE      'P'
#define PTDUMP_DUMP      _IOWR(PTDUMP_BASE, 0, struct ptdump_req)
#define PTDUMP_WRITE     _IOWR(PTDUMP_BASE, 1, struct ptdump_req)
