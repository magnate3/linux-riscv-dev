 

<center><h1>gen topo and flow </h1></center>

[conweave-ns3](https://github.com/conweave-project/conweave-ns3/tree/236a801a00e35de9078635e04acae2f701c21ded)

## gen topo and flow 

```
root@ubuntux86:# tree -L 2  
.
├── autorun.sh
├── config
│   ├── fat_k4_100G_OS2.txt
│   ├── fat_k8_100G_OS2.txt
│   ├── fat_topology_gen.py
│   └── leaf_spine_128_100G_OS2.txt
├── run.py
└── traffic_gen
    ├── AliStorage2019.txt
    ├── custom_rand.py
    ├── custom_rand.pyc
    ├── FbHdp2015.txt
    ├── flow_bins.py
    ├── GoogleRPC2008.txt
    ├── README.md
    ├── Solar2022.txt
    └── traffic_gen.py

2 directories, 15 files
```
执行./autorun.sh     
```
root@ubuntux86:# ./autorun.sh 
Running RDMA Network Load Balancing Simulations (leaf-spine topology) 

---------------------------------- 
TOPOLOGY: leaf_spine_128_100G_OS2 
NETWORK LOAD: 50 
TIME: 0.1 
----------------------------------
 
Run Lossless RDMA experiments... 
Run IRN RDMA experiments... 
Runing all in parallel. Check the processors running on background! 
```

```
root@ubuntux86:# tree -L 2  
.
├── autorun.sh
├── config
│   ├── fat_k4_100G_OS2.txt
│   ├── fat_k8_100G_OS2.txt
│   ├── fat_topology_gen.py
│   ├── L_25.00_CDF_AliStorage2019_N_128_T_100ms_B_100_flow.txt
│   └── leaf_spine_128_100G_OS2.txt
├── mix
│   └── output
├── run.py
└── traffic_gen
    ├── AliStorage2019.txt
    ├── custom_rand.py
    ├── custom_rand.pyc
    ├── FbHdp2015.txt
    ├── flow_bins.py
    ├── GoogleRPC2008.txt
    ├── README.md
    ├── Solar2022.txt
    └── traffic_gen.py

4 directories, 16 files
root@ubuntux86:# 
```

生成L_25.00_CDF_AliStorage2019_N_128_T_100ms_B_100_flow.txt   
生成mix    



## topology_file

默认的 `topology_file` 是 `leaf_spine_128_100G_OS2.txt`，存储在 `config` 文件夹下。

```cpp
// 由 topof 打开
// txt文件内容

// 第一行分别是 节点数node_num 交换机数switch_num 链路数link_num
144 16 192
     
// 第二行分别是 switch_num 个 交换机号
128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143

// 接下来的每一行（共计192行） 5个变量分别是
// 源src 目的dst 速率data_rate 链路延迟link_delay 错误速率error_rate
0 128 100Gbps 1000ns 0
1 128 100Gbps 1000ns 0
2 128 100Gbps 1000ns 0
......
```



## flow_file与从flow的执行流程

`flow_file` 存储在 `config` 文件夹下。

```cpp
// 第一行是流的数量
3
// 后续的每一行 5个变量分别是
// src dst pg maxPacketCount start_time
77 0 3 4287 2.000000003
102 42 3 5328 2.000000237
94 91 3 1956 2.000000438
```


## test
```
mv L_25.00_CDF_AliStorage2019_N_128_T_100ms_B_100_flow.txt  flow.txt 
mv leaf_spine_128_100G_OS2.txt  topology.txt     
```


