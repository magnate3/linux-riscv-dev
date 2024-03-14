obj-m := 
obj-m +=   paging_lowmem.o
#obj-m +=   tlb_test.o
#obj-m +=   page_table_walk_test.o
#obj-m +=   page_dump_test.o
#obj-m +=   page_test.o
#obj-m +=   get_vm_area_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

