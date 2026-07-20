obj-m := hrtime.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

.PHONY:clean insmod rmmod watch lsmod
default:
	    $(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	    $(RM) *.o *.mod.c *.ko *.symvers *.markers *.order
insmod:
	echo "barnard" | sudo -S insmod $(obj-m:%.o=%.ko)
rmmod:
	echo "barnard" | sudo -S rmmod $(obj-m:%.o=%)
lsmod:
	#echo "barnard" | sudo -S lsmod | grep "$(obj-m:%.o=%)"
	lsmod | grep "$(obj-m:%.o=%)"
watch:
	dmesg|tail