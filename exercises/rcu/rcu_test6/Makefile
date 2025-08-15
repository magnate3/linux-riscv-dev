obj-m := 
#obj-m +=   list_rcu_example.o
#obj-m +=   rcu_test1.o
#obj-m +=   rcu_test2.o
#obj-m +=   rcu_test3.o
#obj-m +=   rcu_test5.o
obj-m +=   rcu_test6.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

