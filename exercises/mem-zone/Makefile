obj-m := 
obj-m +=  mem_map_test.o
obj-m +=  get_user_pages_remote_test.o
obj-m +=  vm_file_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

