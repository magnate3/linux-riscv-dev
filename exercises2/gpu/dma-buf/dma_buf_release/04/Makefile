obj-m := 
obj-m +=   exporter-fd-test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc mmap_dmabuf.c  -o mmap_dmabuf
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm mmap_dmabuf

