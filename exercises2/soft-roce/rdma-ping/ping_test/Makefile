EXE = rc_pingpong rping RDMA_RC_example

LDLIBS += -libverbs -lrdmacm
CFLAGS += -std=c99 -D_GNU_SOURCE

default: $(EXE)

rc_pingpong: pingpong.o rc_pingpong.o
rping: rping.o
RDMA_RC_example: RDMA_RC_example.o

clean:
	rm -rf $(EXE) *.o *~
