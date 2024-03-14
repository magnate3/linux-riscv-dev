CC = gcc
RDMA_CORE_PATH = /home/rdma/rdma-core
ifdef RDMA_CORE_PATH
    CFLAGS = -I$(RDMA_CORE_PATH)/build/include
    LDFLAGS = -L$(RDMA_CORE_PATH)/build/lib
    LDFLAGS += -Wl,-rpath -Wl,$(RDMA_CORE_PATH)/build/lib
endif
CFLAGS += -g 
#CFLAGS += -O2 -Wall -W -Werror
LDFLAGS += -libverbs -lrdmacm -lmlx5
TARGETS = dcping

all:
	$(CC) $(CFLAGS) -o $(TARGETS) dcping.c $(LDFLAGS)

clean:
	rm -f $(TARGETS)
