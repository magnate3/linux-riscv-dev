# insmod char_dev_test.ko 
```
[root@centos7 chardevice]# insmod char_dev_test.ko 
[root@centos7 chardevice]# dmesg | tail -n 10
[933344.054630] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[933349.051880] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962696.216940] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962696.222435] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962696.972735] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962697.612099] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962698.716111] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[962714.904150] hns3 0000:7d:00.0 enp125s0f0: L3/L4 error pkt
[1040127.922228] ************** major : 241 
[1040127.926326] ldm_init sucess 
[root@centos7 chardevice]# ls /dev/* | grep 241
241:0
241:1
241:2
[root@centos7 chardevice]# ls /dev/*  -al | grep 241
crw------- 1 root root    241,   0 Sep 10 06:43 /dev/ldm_led_0
crw------- 1 root root    241,   1 Sep 10 06:43 /dev/ldm_led_1
crw------- 1 root root    241,   2 Sep 10 06:43 /dev/ldm_led_2
[root@centos7 chardevice]# rmmod  char_dev_test.ko 
[root@centos7 chardevice]# ls /dev/*  -al | grep 241
```