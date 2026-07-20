
# test.c

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/huagepage/pic.png)

#  hugepage-addr 
```
[root@centos7 hugepage-addr]# touch /mnt/huge
[root@centos7 hugepage-addr]# mount -t hugetlbfs nodev /mnt/huge
[root@centos7 hugepage-addr]# cat /proc/meminfo  | grep -i huge
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
HugePages_Total:      64
HugePages_Free:       64
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:     524288 kB
[root@centos7 hugepage-addr]# ./hugepage-addr 
Fail to open hugepage memory file: Is a directory
[root@centos7 hugepage-addr]# 
[root@centos7 hugepage-addr]# mount | grep huge
cgroup on /sys/fs/cgroup/hugetlb type cgroup (rw,nosuid,nodev,noexec,relatime,hugetlb)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=512M)
nodev on /mnt/huge type hugetlbfs (rw,relatime,pagesize=512M)
[root@centos7 hugepage-addr
```

**change #define HUGEPAGE_FILE "/mnt/huge"  to #define HUGEPAGE_FILE "/mnt/huge/test1"**

```
[root@centos7 hugepage-addr]# umount /mnt/huge
[root@centos7 hugepage-addr]#  rm /mnt/huge/ -rf
[root@centos7 hugepage-addr]# mkdir /mnt/huge
[root@centos7 hugepage-addr]# make
[root@centos7 hugepage-addr]# ./hugepage-addr 
Zero page frame number
Virtual address is 0xffffb0a30000
Physical address is 0
[root@centos7 hugepage-addr]# 
```


##   Bus error

```
[root@centos7 hugepage]# cat  /sys/devices/system/node/node*/hugepages/hugepages-524288kB/nr_hugepages
16
16
16
16
[root@centos7 hugepage]# 
```
***the problem is not mount /mnt/huge/***
```
[root@centos7 hugepage-addr]# mount -t hugetlbfs nodev /mnt/huge
[root@centos7 hugepage-addr]# ./hugepage-addr 
Virtual address is 0x400000000000
Physical address is 35581119692800
97
[root@centos7 hugepage-addr]# umount /mnt/huge/
[root@centos7 hugepage-addr]# ./hugepage-addr 
Zero page frame number
Virtual address is 0xffff82190000
Physical address is 0
Bus error
[root@centos7 hugepage-addr]# mount | grep -i huge
cgroup on /sys/fs/cgroup/hugetlb type cgroup (rw,nosuid,nodev,noexec,relatime,hugetlb)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=512M)
[root@centos7 hugepage-addr]# mount -t hugetlbfs nodev /mnt/huge
[root@centos7 hugepage-addr]# ./hugepage-addr 
Virtual address is 0x400000000000
Physical address is 35581119692800
97
[root@centos7 hugepage-addr]# 
```

# hugepage-user

## insmod  hugepage-driver.ko 

```
[root@centos7 hugepage-user]# dmesg | tail -n 10
[1723029.142962] x3 : 0000000000000000 x2 : 0000000000000001 
[1723029.148430] x1 : 0000000000000061 x0 : 0000ffff82190000 
[1723226.670943] Install hugepage-driver
[1723254.790588] dev_open
[1723254.793071] dev_release
[1723308.966580] dev_open
[1723308.992198] The physical address is 35581119692800
[1723308.997144] The virtual address is ffffa05c60000000
[1723309.002178] The memory length is 1048576
[1723309.010894] dev_release
[root@centos7 hugepage-user]# 
```
## ./hugepage-user 
```
[root@centos7 hugepage-user]# ./hugepage-user 
Virtual address is 0x400000000000
Physical address is 35581119692800
The value of the first byte is: 1
All the values are correct
```

# test2

```
echo 64 > /sys/kernel/mm/hugepages/hugepages-524288kB/nr_hugepages
[root@centos7 hugepage]# gcc  test2.c  -o test2
[root@centos7 hugepage]# ./test2
Returned address is 0x400000000000
First hex is 0
First hex is 3020100
input char 

Returned address is 0x400180000000
First hex is 0
First hex is 3020100
input char 

munmap addr1 0x400000000000, addr2 0x400180000000
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/huagepage/fault1.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/huagepage/fault2.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/huagepage/fault3.png)