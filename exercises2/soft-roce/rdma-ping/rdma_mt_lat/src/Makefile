CC = gcc
#CFLAGS += -g -Wall
CFLAGS += -g -O2 -Wall

# -Werror
#CFLAGS += -Werror -Wswitch-enum -W
#CFLAGS += -pedantic -ansi
LDFLAGS += -libverbs -lpthread
OBJECTS = rdma_mt_send_lat.o rdma_thread.o rdma_resource.o rdma_ops.o rdma_utils.o
TARGETS = rdma_mt_send_lat

all: $(TARGETS)

$(TARGETS): $(OBJECTS)
	$(CC) $^ -o $@ $(LDFLAGS)

rdma_mt_send_lat.o: rdma_mt_send_lat.c rdma_mt.h
	$(CC) -c $(CFLAGS) $<

rdma_thread.o: rdma_thread.c rdma_mt.h
	$(CC) -c $(CFLAGS) $<

rdma_resource.o: rdma_resource.c rdma_mt.h
	$(CC) -c $(CFLAGS) $<

rdma_ops.o: rdma_ops.c rdma_mt.h
	$(CC) -c $(CFLAGS) $<

rdma_utils.o: rdma_utils.c rdma_utils.h
	$(CC) -c $(CFLAGS) $<
	
clean:
	rm -f $(OBJECTS) $(TARGETS)

# for install/uninstall roles
#ifneq (,$(MLXTST_SRCROOT))
#include $(MLXTST_SRCROOT)/scripts/mlx_test_inc.mk
#endif

