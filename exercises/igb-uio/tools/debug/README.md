# ./pcimem r 05:00.0 0 0 4 0
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/pci_mem.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/hinic_res.png)


# igb_pci_mem
```
[root@centos7 user]# ./igb_pci_mem 
The device address 0x80007b00000 (lenth 0000000000020000l)
can be accessed over
logical address 0xffff9b000000
@0x00000000 = 0x80000000
@0x00000004 = 0xc1502200
[root@centos7 user]# dmesg | grep 'igb uio driver'
[1316119.473399] igb_uio: in igb uio driver attr0 80000000, and attr1 c1502200 
[1479208.433365] igb_uio: in igb uio driver attr0 80000000, and attr1 c1502200 
[1481070.027542] igb_uio: in igb uio driver attr0 80000000, and attr1 c1502200 
[1481070.101672] igb_uio: in igb uio driver attr0 80000000, and attr1 c1502200 
[root@centos7 user]# 
```

```
[root@centos7 igb-uio]# cat  /sys/class/uio/uio0/maps/map*/addr
0x0000080007b00000
0x0000080008a20000
0x0000080000200000
[root@centos7 igb-uio]# cat  /sys/class/uio/uio0/maps/map*/name
BAR0
BAR2
BAR4
[root@centos7 igb-uio]# cat  /sys/class/uio/uio0/maps/map*/offset
0x0
0x0
0x0
[root@centos7 igb-uio]# cat  /sys/class/uio/uio0/maps/map*/size
0x0000000000020000
0x0000000000008000
0x0000000000100000
[root@centos7 igb-uio]# 

[root@centos7 user]# cat  /sys/bus/pci/devices/0000:05:00.0/resource
0x0000080007b00000 0x0000080007b1ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080008a20000 0x0000080008a27fff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080000200000 0x00000800002fffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x00000000e9200000 0x00000000e92fffff 0x0000000000046200
0x0000080007b20000 0x000008000829ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x00000800082a0000 0x0000080008a1ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080000300000 0x0000080007afffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
[root@centos7 user]# 
```
## attention

```
  uint64_t *uio_addr; 
  void *access_address; //  not  uint64_t *access_address;
```

## sizeof(void *)  and  void * +1

```
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

int main(void)
{
        void *  addr = 0x0000080007b00000;
        uint64_t num = 0x0000080007b00000;
        printf("%lu\n", num);             //十进制输出
        printf("0x%"PRIx64"\n", num);     //十六进制输出
        printf("0x%016lx\n", num);        //十六进制输出
        //no long
        printf("0x%016x\n", num);        //十六进制输出
        //no zero
        printf("0x%16lx\n", num);        //十六进制输出
        printf("addr size %d and void size %d \n", sizeof(addr), sizeof(void));        //十六进制输出  
        printf("0x%016lx and addr +1 0x%016lx \n", addr, addr +1);        //十六进制输出
}
```

```
[root@centos7 user]# ./test3
8796221997056
0x80007b00000
0x0000080007b00000
0x0000000007b00000
0x     80007b00000
addr size 8 and void size 1 
0x0000080007b00000 and addr +1 0x0000080007b00001 
[root@centos7 user]# 
```
