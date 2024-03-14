
# insmod  zone_test1.ko

```
[root@centos7 zone]# insmod  zone_test1.ko 
[root@centos7 zone]# dmesg | tail -n 10
[142462.678185] z->zone->name: Normal
[142462.681577] z->zone->name: Normal
[root@centos7 zone]# 
```

# insmod  zone_test2.ko 

```
[142961.595659] NODES_SHIFT = 2, MAX_NUMNODES = 4
[142961.600093] zone->name: DMA
[142961.602961]         [0] nr_free: 12
[142961.605915]         [1] nr_free: 21
[142961.608875]         [2] nr_free: 8
[142961.611744]         [3] nr_free: 12
[142961.614698]         [4] nr_free: 12
[142961.617656]         [5] nr_free: 13
[142961.620611]         [6] nr_free: 11
[142961.623565]         [7] nr_free: 6
[142961.626431]         [8] nr_free: 6
[142961.629308]         [9] nr_free: 5
[142961.632177]         [10] nr_free: 4
[142961.635131]         [11] nr_free: 2
[142961.638091]         [12] nr_free: 1
[142961.641046]         [13] nr_free: 0
[142961.644000] zone->name: Normal
[142961.647133]         [0] nr_free: 48
[142961.650088]         [1] nr_free: 8
[142961.652955]         [2] nr_free: 17
[142961.655909]         [3] nr_free: 5
[142961.658781]         [4] nr_free: 5
[142961.661649]         [5] nr_free: 4
[142961.664516]         [6] nr_free: 2
[142961.667389]         [7] nr_free: 1
[142961.670257]         [8] nr_free: 4
[142961.673124]         [9] nr_free: 2
[142961.675991]         [10] nr_free: 3
[142961.678950]         [11] nr_free: 1
[142961.681904]         [12] nr_free: 3
[142961.684858]         [13] nr_free: 232
```
# insmod  zone_test3.ko 

```
[root@centos7 zone]# insmod  zone_test3.ko 
[root@centos7 zone]# dmesg | tail -n 100
```

```
[53807.946738]  ************************ ZONE NAME Normal
[53807.951914]  ZONE NAME Normal, ZONE 1 free_area[0].nr_free = 0x38
[53807.957979]  ZONE 1 free_area[0].free_list[Unmovable] = 0x13
[53807.963616]  ZONE 1 free_area[0].free_list[Reclaimable] = 0x0
[53807.969336]  ZONE 1 free_area[0].free_list[Movable] = 0x43
[53807.974801]  ZONE 1 free_area[0].free_list[Isolate] = 0x0
[53807.980181]  ZONE NAME Normal, ZONE 1 free_area[1].nr_free = 0x2c
[53807.986247]  ZONE 1 free_area[1].free_list[Unmovable] = 0x17
[53807.991885]  ZONE 1 free_area[1].free_list[Reclaimable] = 0x0
[53807.997605]  ZONE 1 free_area[1].free_list[Movable] = 0x27
[53808.003071]  ZONE 1 free_area[1].free_list[Isolate] = 0x0
[53808.008445]  ZONE NAME Normal, ZONE 1 free_area[2].nr_free = 0x12
[53808.014515]  ZONE 1 free_area[2].free_list[Unmovable] = 0x4
[53808.020069]  ZONE 1 free_area[2].free_list[Reclaimable] = 0x1
[53808.025789]  ZONE 1 free_area[2].free_list[Movable] = 0x13
[53808.031255]  ZONE 1 free_area[2].free_list[Isolate] = 0x0
[53808.036628]  ZONE NAME Normal, ZONE 1 free_area[3].nr_free = 0x10
[53808.042699]  ZONE 1 free_area[3].free_list[Unmovable] = 0x1
[53808.048247]  ZONE 1 free_area[3].free_list[Reclaimable] = 0x1
[53808.053972]  ZONE 1 free_area[3].free_list[Movable] = 0x14
[53808.059433]  ZONE 1 free_area[3].free_list[Isolate] = 0x0
[53808.064811]  ZONE NAME Normal, ZONE 1 free_area[4].nr_free = 0xa
[53808.070795]  ZONE 1 free_area[4].free_list[Unmovable] = 0x1
[53808.076341]  ZONE 1 free_area[4].free_list[Reclaimable] = 0x1
[53808.082066]  ZONE 1 free_area[4].free_list[Movable] = 0x8
[53808.087440]  ZONE 1 free_area[4].free_list[Isolate] = 0x0
[53808.092822]  ZONE NAME Normal, ZONE 1 free_area[5].nr_free = 0x6
[53808.098800]  ZONE 1 free_area[5].free_list[Unmovable] = 0x1
[53808.104354]  ZONE 1 free_area[5].free_list[Reclaimable] = 0x1
[53808.110081]  ZONE 1 free_area[5].free_list[Movable] = 0x4
[53808.115456]  ZONE 1 free_area[5].free_list[Isolate] = 0x0
[53808.120835]  ZONE NAME Normal, ZONE 1 free_area[6].nr_free = 0x4
[53808.126814]  ZONE 1 free_area[6].free_list[Unmovable] = 0x0
[53808.132366]  ZONE 1 free_area[6].free_list[Reclaimable] = 0x1
[53808.138086]  ZONE 1 free_area[6].free_list[Movable] = 0x3
[53808.143465]  ZONE 1 free_area[6].free_list[Isolate] = 0x0
[53808.148839]  ZONE NAME Normal, ZONE 1 free_area[7].nr_free = 0x0
[53808.154823]  ZONE 1 free_area[7].free_list[Unmovable] = 0x0
[53808.160377]  ZONE 1 free_area[7].free_list[Reclaimable] = 0x0
[53808.166097]  ZONE 1 free_area[7].free_list[Movable] = 0x0
[53808.171477]  ZONE 1 free_area[7].free_list[Isolate] = 0x0
[53808.176850]  ZONE NAME Normal, ZONE 1 free_area[8].nr_free = 0x2
[53808.182836]  ZONE 1 free_area[8].free_list[Unmovable] = 0x0
[53808.188382]  ZONE 1 free_area[8].free_list[Reclaimable] = 0x1
[53808.194108]  ZONE 1 free_area[8].free_list[Movable] = 0x1
[53808.199481]  ZONE 1 free_area[8].free_list[Isolate] = 0x0
[53808.204861]  ZONE NAME Normal, ZONE 1 free_area[9].nr_free = 0x1
[53808.210846]  ZONE 1 free_area[9].free_list[Unmovable] = 0x1
[53808.216395]  ZONE 1 free_area[9].free_list[Reclaimable] = 0x0
[53808.222120]  ZONE 1 free_area[9].free_list[Movable] = 0x0
[53808.227494]  ZONE 1 free_area[9].free_list[Isolate] = 0x0
[53808.232873]  ZONE NAME Normal, ZONE 1 free_area[10].nr_free = 0x1
[53808.238939]  ZONE 1 free_area[10].free_list[Unmovable] = 0x1
[53808.244577]  ZONE 1 free_area[10].free_list[Reclaimable] = 0x0
[53808.250392]  ZONE 1 free_area[10].free_list[Movable] = 0x0
[53808.255853]  ZONE 1 free_area[10].free_list[Isolate] = 0x0
[53808.261320]  ZONE NAME Normal, ZONE 1 free_area[11].nr_free = 0x3
[53808.267386]  ZONE 1 free_area[11].free_list[Unmovable] = 0x1
[53808.273024]  ZONE 1 free_area[11].free_list[Reclaimable] = 0x1
[53808.278831]  ZONE 1 free_area[11].free_list[Movable] = 0x1
[53808.284297]  ZONE 1 free_area[11].free_list[Isolate] = 0x0
[53808.289765]  ZONE NAME Normal, ZONE 1 free_area[12].nr_free = 0x2
[53808.295831]  ZONE 1 free_area[12].free_list[Unmovable] = 0x1
[53808.301469]  ZONE 1 free_area[12].free_list[Reclaimable] = 0x1
[53808.307276]  ZONE 1 free_area[12].free_list[Movable] = 0x0
[53808.312743]  ZONE 1 free_area[12].free_list[Isolate] = 0x0
[53808.318204]  ZONE NAME Normal, ZONE 1 free_area[13].nr_free = 0xf3
[53808.324362]  ZONE 1 free_area[13].free_list[Unmovable] = 0x0
[53808.330001]  ZONE 1 free_area[13].free_list[Reclaimable] = 0x0
[53808.335839]  ZONE 1 free_area[13].free_list[Movable] = 0x243
[53808.341476]  ZONE 1 free_area[13].free_list[Isolate] = 0x0
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mem_node/zone/buddy.png)

[kernel_modules/misc/pglist_data.c](https://github.com/saraboju-umaraju/kernel_modules/blob/b64c6a934c13f3cec8a79481ed36518fa4e84973/misc/pglist_data.c)