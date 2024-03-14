#CFLAGS := -m64 -O2 -g -pthread -D_GNU_SOURCE -D_REENTRANT -W
#LDFLAGS := -m64 -lm
CFLAGS :=  -O2 -g -pthread -D_GNU_SOURCE -D_REENTRANT -W -I ../
LDFLAGS :=  -lm
NVME_PROGS =  test_read
ALL_PROGS := $(NVME_PROGS)
default: $(ALL_PROGS)
	gcc mmap_write.c -o mmap_write -W -I ../
clean:
	rm -f $(ALL_PROGS) *.o
	rm mmap_write
.PHONY: default clean 
