#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

CFLAGS  = -Wall -Werror -g -O2 -I../include
LDFLAGS = -libverbs -lrdmacm
RM      = rm -f

all: rc_read_fc

rc_read_fc.o: ../include/list.h

rc_read_fc: rc_read_fc.o
	$(CC) rc_read_fc.o -o rc_read_fc $(LDFLAGS)
	
clean:
	$(RM) rc_read_fc.o rc_read_fc
	
