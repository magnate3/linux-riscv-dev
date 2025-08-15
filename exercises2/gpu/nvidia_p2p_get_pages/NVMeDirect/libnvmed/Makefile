#include ../../common/make.config

SRC = nvmed.c pinpool.c fifo.c pinpool.h nvmed.h fifo.h nvme_dev.c nvme_dev.h 
#filemap.c filemap.h

OBJS = nvmed.o pinpool.o nvme_dev.o fifo.o 

OUTPUT = libnvmed.so libnvmed.a

CFLAGS += -O3 -fPIC -D_GNU_SOURCE -g -I/usr/local/include -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -L/usr/local/cuda-8.0/lib64
LD_FLAGS += -lcuda -lcudart 
all: $(OUTPUT)

libnvmed.so: $(OBJS) 
	$(CC) $(CFLAGS) -shared -Wl,-soname,libnvmed.so.1 -o libnvmed.so $(OBJS) -lc $(LD_FLAGS)

libnvmed.a: $(OBJS)
	$(AR) rcs libnvmed.a $(OBJS)
	
nvmed.o: nvmed.c nvmed.h pinpool.o nvme_dev.o fifo.o filemap.o
	$(CC) -c -fPIC $(CFLAGS) nvmed.c pinpool.o fifo.o nvme_dev.o filemap.o -o nvmed.o $(LD_FLAGS)

pinpool.o: pinpool.c pinpool.h fifo.o
	$(CC) -c $(CFLAGS) pinpool.c fifo.o -o pinpool.o $(LD_FLAGS)

fifo.o: fifo.c fifo.h 
	$(CC) -c $(CFLAGS) fifo.c -o fifo.o $(LD_FLAGS)

filemap.o: filemap.c filemap.h nvme_dev.o
	$(CC) -c $(CFLAGS) filemap.c nvme_dev.o -o filemap.o $(LD_FLAGS)

nvme_dev.o: nvme_dev.c nvme_dev.h 
	$(CC) -c $(CFLAGS) nvme_dev.c -o nvme_dev.o $(LD_FLAGS)


install: 
	mkdir -p /usr/local/lib
	mkdir -p /usr/include/nvmed
	cp *.so /usr/local/lib
	cp *.a /usr/local/lib
	cp *.h /usr/include/nvmed
		
clean: $(SRC)
	rm -f *.so *.a *.o
