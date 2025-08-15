#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

CFLAGS  = -Wall -Werror -g -O2 -I../include
LDFLAGS = -libverbs -lrdmacm -lmlx5
RM      = rm -f

all: dc_read_fc

dc_read_fc.o: ../include/list.h

dc_read_fc: dc_read_fc.o
	$(CC) dc_read_fc.o -o dc_read_fc $(LDFLAGS)
	
clean:
	$(RM) dc_read_fc.o dc_read_fc
	
