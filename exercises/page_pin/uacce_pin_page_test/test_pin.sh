#!/bin/sh
device=page_pin_test
rm -f /dev/${device}
rmmod page_pin_test.ko
insmod page_pin_test.ko
mknod --mode=666 /dev/${device} c  `grep page_pin_test /proc/devices | awk '{print $1;}'` 0 &&  ls -al /dev/${device}
#./test_pin &
#PID=`pidof test_pin`
#
#for i in `seq 10`
#do
#	#numastat -p $PID
#	sleep 2
#done
#
#kill -9 $PID
rmmod page_pin_test.ko
rm -f /dev/${device}
