obj-m := mybrd_test.o
obj-m += mybrd_bio_mode_test.o
#obj-m := null_blk.o
KERN_SOURCE := /lib/modules/$(shell uname -r)/build

default:
		$(MAKE) -C $(KERN_SOURCE) M=$(PWD) modules
clean:
		$(MAKE) -C $(KERN_SOURCE) M=$(PWD) clean
