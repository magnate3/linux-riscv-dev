insmod nat64_device.ko 
ip a add 2001:db8::a0a:6751/96 dev nat64
ip l set nat64 up
ip a add 10.10.103.82/24 dev nat64
