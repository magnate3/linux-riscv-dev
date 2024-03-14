insmod  net_test.ko 
ip a add 10.10.10.10 dev eth0
ip link set eth0 up
ping 10.10.10.10 -c 1 -s 64
rmmod net_test
