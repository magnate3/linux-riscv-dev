
obj-m := 
obj-m +=    kernel_netlink_test.o

MKD := mkdir -p $(BUILD_DIR)

all: kernel user

kernel:
	make -C /lib/modules/$(shell uname -r)/build \
        M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
        M=$(PWD) clean
	rm user_netlink_test
user:
	gcc -g  user_netlink_test.c -o user_netlink_test 
	gcc -g  nlcore.c user2.c -o test 

.PHONY: kernel user clean
