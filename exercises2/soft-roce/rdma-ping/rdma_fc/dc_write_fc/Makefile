#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

CFLAGS  = -Wall -Werror -g -O2 -I../include
LDFLAGS = -libverbs -lrdmacm -lmlx5
RM      = rm -f

all: dc_write_fc

dc_write_fc.o: ../include/list.h

dc_write_fc: dc_write_fc.o
	$(CC) dc_write_fc.o -o dc_write_fc $(LDFLAGS)
	
clean:
	$(RM) dc_write_fc.o dc_write_fc
	
