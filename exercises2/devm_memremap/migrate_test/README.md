

```
root@ubuntux86:# insmod  mmu_test.ko 
root@ubuntux86:# ./mmap_test 
addr: 0x7f0ac0ad7000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
virt addr 0x55cf5699d000, phy addr of ptr  0x170312000 
Zero page frame number
after migrate, phy addr of ptr 0x0 
Zero page frame number
after migrate and memecpy again , phy addr of ptr 0x0 
Zero page frame number
after migrate and memecpy again , phy addr of ptr 0x0 

root@ubuntux86:# dmesg | tail -n 50
[29826.822314] (page ZONE: Device)
[29826.822315] (page ZONE: Device)
[29826.822315] (page ZONE: Device)
[29826.822316] (page ZONE: Device)
[29826.822316] (page ZONE: Device)
[29826.822317] (page ZONE: Device)
[29826.822317] (page ZONE: Device)
[29826.822318] (page ZONE: Device)
[29826.822318] (page ZONE: Device)
[29826.822319] (page ZONE: Device)
[29826.822319] (page ZONE: Device)
[29826.822320] (page ZONE: Device)
[29826.822320] (page ZONE: Device)
[29826.822321] (page ZONE: Device)
[29826.822321] (page ZONE: Device)
[29826.822322] (page ZONE: Device)
[29826.822322] (page ZONE: Device)
[29826.822323] (page ZONE: Device)
[29826.822323] (page ZONE: Device)
[29826.822324] (page ZONE: Device)
[29826.822324] (page ZONE: Device)
[29826.822325] (page ZONE: Device)
[29826.822326] (page ZONE: Device)
[29826.822326] (page ZONE: Device)
[29826.822326] (page ZONE: Device)
[29826.822327] (page ZONE: Device)
[29826.822328] (page ZONE: Device)
[29826.822328] (page ZONE: Device)
[29826.822328] (page ZONE: Device)
[29826.822329] (page ZONE: Device)
[29826.822330] (page ZONE: Device)
[29826.822330] (page ZONE: Device)
[29826.822331] (page ZONE: Device)
[29826.822331] (page ZONE: Device)
[29826.822332] (page ZONE: Device)
[29826.822332] (page ZONE: Device)
[29826.822333] (page ZONE: Device)
[29826.822333] (page ZONE: Device)
[29826.822334] (page ZONE: Device)
[29826.822334] (page ZONE: Device)
[29826.822335] (page ZONE: Device)
[29826.822448] (pgmap size: 32767 pages)
[29826.822449] Hello modules on myMMU
[29829.897573] myMMU notifier: change_pte
[29829.897871] dmirror_migrate_alloc_and_copy call copy highpage  ,dpage @ 00000000debaedae, spage @ 0000000075c8f937 
[29829.897887] src and dts page are equal 
[29829.897889] buf is krishna 
[29829.897958] dmirror_devmem_fault addr 0x55cf5699d000 
[29829.898008] ------------- pte present 
[29829.898012]  not swap pte,page frame struct is @ 0000000075c8f937, and user paddr 0x170312000, virt addr 0x55cf5699d000 
root@ubuntux86:# 
```