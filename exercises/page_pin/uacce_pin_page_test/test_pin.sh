#!/bin/sh
device=uacce_pin_page
type=c
major=42
minor=0
rm -f /dev/${device}
mknod /dev/${device} $type $major $minor && ls -al /dev/${device}
insmod page_pin_test.ko
./test_pin &
PID=`pidof test_pin`

for i in `seq 10`
do
	numastat -p $PID
	sleep 2
done

kill -9 $PID
rm -f /dev/${device}
rmmod page_pin_test.ko
