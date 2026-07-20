obj-m := 
obj-m +=   dlmalloc_test.o
dlmalloc_test-y :=  dlmalloc_mod.o mspace_allocator.o
#dlmalloc_test-y :=  dlmalloc_mod.o malloc.o
EXTRA_CFLAGS += -DUSE_DLMALLOC
EXTRA_CFLAGS += -DUSE_DL_PREFIX
all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

