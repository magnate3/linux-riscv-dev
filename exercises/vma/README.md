
#  echo 'listvma' >  /proc/mtest
```
[root@centos7 vma]# dmesg | tail -n 100
[86937.080381] page_cache_get_init
[86937.083514] befor page_cache_get pages->_count :8589934591, 1
[86937.089243] after page_cache_get pages->_count :12884901887, 2
[86937.095051] after page_cache_release pages->_count :8589934591, 1
[87016.678295] page_cache_get_exit
[98661.147823] create the filename mtest mtest_init sucess  
[98708.163985] mtest_write  ………..  
[98708.167728] The current process is bash
[98708.171546] mtest_dump_vma_list
[98708.174683] VMA 0x400000-0x4e0000 
[98708.174684] READ 
[98708.178076] EXEC 

[98708.183397] VMA 0x4e0000-0x4f0000 
[98708.183398] READ 

[98708.190193] VMA 0x4f0000-0x500000 
[98708.190194] WRITE 
[98708.193580] READ 

[98708.198990] VMA 0x26060000-0x260c0000 
[98708.198991] WRITE 
[98708.202723] READ 

[98708.208136] VMA 0xffffa3390000-0xffffa33b0000 
[98708.208137] WRITE 
[98708.212560] READ 

[98708.217970] VMA 0xffffa33b0000-0xffffa9c70000 
[98708.217971] READ 

[98708.225795] VMA 0xffffa9c70000-0xffffa9c80000 
[98708.225796] READ 
[98708.230224] EXEC 

[98708.235545] VMA 0xffffa9c80000-0xffffa9c90000 
[98708.235545] READ 

[98708.243375] VMA 0xffffa9c90000-0xffffa9ca0000 
[98708.243375] WRITE 
[98708.247805] READ 

[98708.253211] VMA 0xffffa9ca0000-0xffffa9e10000 
[98708.253212] READ 
[98708.257639] EXEC 

[98708.262959] VMA 0xffffa9e10000-0xffffa9e20000 
[98708.262960] READ 

[98708.270790] VMA 0xffffa9e20000-0xffffa9e30000 
[98708.270791] WRITE 
[98708.275213] READ 

[98708.280623] VMA 0xffffa9e30000-0xffffa9e40000 
[98708.280624] READ 
[98708.285047] EXEC 

[98708.290372] VMA 0xffffa9e40000-0xffffa9e50000 
[98708.290373] READ 

[98708.298203] VMA 0xffffa9e50000-0xffffa9e60000 
[98708.298204] WRITE 
[98708.302626] READ 

[98708.308036] VMA 0xffffa9e60000-0xffffa9e90000 
[98708.308038] READ 
[98708.312460] EXEC 

[98708.317785] VMA 0xffffa9e90000-0xffffa9ea0000 
[98708.317785] READ 

[98708.325610] VMA 0xffffa9ea0000-0xffffa9eb0000 
[98708.325611] WRITE 
[98708.330038] READ 

[98708.335444] VMA 0xffffa9eb0000-0xffffa9ec0000 
[98708.335444] READ 

[98708.343276] VMA 0xffffa9ec0000-0xffffa9ed0000 
[98708.343278] READ 

[98708.351108] VMA 0xffffa9ed0000-0xffffa9ee0000 
[98708.351108] READ 
[98708.355531] EXEC 

[98708.360855] VMA 0xffffa9ee0000-0xffffa9f00000 
[98708.360856] READ 
[98708.365278] EXEC 

[98708.370603] VMA 0xffffa9f00000-0xffffa9f10000 
[98708.370604] READ 

[98708.378433] VMA 0xffffa9f10000-0xffffa9f20000 
[98708.378434] WRITE 
[98708.382857] READ 

[98708.388267] VMA 0xffffc8fd0000-0xffffc9000000 
[98708.388268] WRITE 
[98708.392691] READ 

[root@centos7 vma]# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm1.png)

#  echo "findvma0xb7f2b001" > /proc/mtest 

```
[root@centos7 vma]# echo "findvma0xb7f2b001" > /proc/mtest 
[root@centos7 vma]# dmesg | tail -n 10
[98708.378434] WRITE 
[98708.382857] READ 

[98708.388267] VMA 0xffffc8fd0000-0xffffc9000000 
[98708.388268] WRITE 
[98708.392691] READ 

[99286.198310] mtest_write  ………..  
[99286.202050] mtest_find_vma
[99286.204746] no vma found for b7f2b001
[root@centos7 vma]# 
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm2.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm3.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm4.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm5.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm51.png)



#  findpage

```
[root@centos7 vma]# echo "findpage0xffffc8fd0000" > /proc/mtest 
[root@centos7 vma]# echo "findpage0x4e0000" > /proc/mtest 
[root@centos7 vma]# echo "findpage0xffffa9f11000" > /proc/mtest 
[root@centos7 vma]# dmesg | tail -n 10
[99857.902661] mtest_write_val
[99857.905443] page not found  for 0xffffc8fd0000
[99871.443890] mtest_write  ………..  
[99871.447631] mtest_write_val
[99871.450413] page  found  for 0x4e0000
[99871.454058] find  0x4e0000 to kernel address 0xffffa05e75210000
[99894.236396] mtest_write  ………..  
[99894.240130] mtest_write_val
[99894.242912] page  found  for 0xffffa9f11000
[99894.247091] find  0xffffa9f11000 to kernel address 0xffffa03fe8e21000
[root@centos7 vma]# dmesg | tail -n 20
[99395.125746] mtest_write  ………..  
[99395.129487] mtest_find_vma
[99395.132184] found vma 0xffffa9f10000-0xffffa9f20000 flag 100873 for addr 0xffffa9f11000
[99636.020279] mtest_write  ………..  
[99636.024019] mtest_find_vma
[99636.026715] found vma 0x4e0000-0x4f0000 flag 100871 for addr 0x4e0000
[99768.517745] mtest_write  ………..  
[99768.521497] mtest_find_vma
[99768.524201] found vma 0xffffc8fd0000-0xffffc9000000 flag 100173 for addr 0xffffc8fd0000
[99857.898924] mtest_write  ………..  
[99857.902661] mtest_write_val
[99857.905443] page not found  for 0xffffc8fd0000
[99871.443890] mtest_write  ………..  
[99871.447631] mtest_write_val
[99871.450413] page  found  for 0x4e0000
[99871.454058] find  0x4e0000 to kernel address 0xffffa05e75210000
[99894.236396] mtest_write  ………..  
[99894.240130] mtest_write_val
[99894.242912] page  found  for 0xffffa9f11000
[99894.247091] find  0xffffa9f11000 to kernel address 0xffffa03fe8e21000
[root@centos7 vma]# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vm6.png)


# linux内核那些事之Sparse内存模型初始化

```
[root@centos7 boot]# grep CONFIG_SPARSEMEM_EXTREME  config-4.14.0-115.el7a.0.1.aarch64
CONFIG_SPARSEMEM_EXTREME=y
[root@centos7 boot]# grep CONFIG_SPARSEMEM_VMEMMAP config-4.14.0-115.el7a.0.1.aarch64
CONFIG_SPARSEMEM_VMEMMAP_ENABLE=y
CONFIG_SPARSEMEM_VMEMMAP=y
[root@centos7 boot]# 
```