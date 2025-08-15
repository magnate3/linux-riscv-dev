/*
 * Copyright 2016 Ecole Polytechnique Federale Lausanne (EPFL)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St., Fifth Floor, Boston, MA  02110-1301, USA.
 */

struct pci_loc {
	int domain;
	unsigned int bus;
	unsigned int slot;
	unsigned int func;
};

struct args_enable {
	struct pci_loc pci_loc;
};

#define PCIDMA_ENABLE _IOR('a', 0x01, struct args_enable)
