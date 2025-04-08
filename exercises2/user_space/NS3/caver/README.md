
# run conweave 

```
wget https://www.nsnam.org/releases/ns-allinone-3.19.tar.bz2
tar -xvf ns-allinone-3.19.tar.bz2
cd ns-allinone-3.19
rm -rf ns-3.19
git clone https://github.com/conweave-project/conweave-ns3.git ns-3.19
cd ns-3.19
./waf configure --build-profile=optimized
./waf
```

# run caver 

+ gcc    

```
gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

```
wget https://www.nsnam.org/releases/ns-allinone-3.19.tar.bz2
tar -xvf ns-allinone-3.19.tar.bz2
cd ns-allinone-3.19
rm -rf ns-3.19
git clone https://github.com/denght23/CAVER.git ns-3.19
cd ns-3.19
./waf configure --build-profile=optimized
./waf
```


```
root@ubuntux86:# ./autorun.sh
Running RDMA Network Load Balancing Simulations (leaf-spine topology) 

---------------------------------- 
TOPOLOGY: fat_k4_100G_OS2 
NETWORK LOAD: 52 
TIME: 0.1 
----------------------------------
 
Run Lossless RDMA experiments... 
Runing all in parallel. Check the processors running on background! 
root@ubuntux86:# ps -elf | grep waf
0 S root       75811   75788  0  80   0 -   654 do_wai 15:28 pts/0    00:00:00 sh -c ./waf --run 'scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/261339713/config.txt' > /work/ns-allinone-3.19/ns-3.19/mix/output/261339713/config.log 2>&1
0 S root       75812   75811  0  80   0 - 15396 do_wai 15:28 pts/0    00:00:00 python ./waf --run scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/261339713/config.txt
0 S root       75836   75815  0  80   0 -   654 do_wai 15:28 pts/0    00:00:00 sh -c ./waf --run 'scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/511323115/config.txt' > /work/ns-allinone-3.19/ns-3.19/mix/output/511323115/config.log 2>&1
0 S root       75837   75836  0  80   0 - 15396 do_wai 15:28 pts/0    00:00:00 python ./waf --run scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/511323115/config.txt
0 S root       75861   75840  0  80   0 -   654 do_wai 15:28 pts/0    00:00:00 sh -c ./waf --run 'scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/599272230/config.txt' > /work/ns-allinone-3.19/ns-3.19/mix/output/599272230/config.log 2>&1
0 S root       75862   75861  0  80   0 - 15395 do_wai 15:28 pts/0    00:00:00 python ./waf --run scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/599272230/config.txt
0 S root       75887   75866  0  80   0 -   654 do_wai 15:29 pts/0    00:00:00 sh -c ./waf --run 'scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/950743702/config.txt' > /work/ns-allinone-3.19/ns-3.19/mix/output/950743702/config.log 2>&1
0 S root       75888   75887  0  80   0 - 15397 do_wai 15:29 pts/0    00:00:00 python ./waf --run scratch/network-load-balance /work/ns-allinone-3.19/ns-3.19/mix/output/950743702/config.txt
0 S root       75910    3204  0  80   0 -  3028 -      15:31 pts/0    00:00:00 grep --color=auto waf
```

```
bash autorun.sh 
Running RDMA Network Load Balancing Simulations (leaf-spine topology) 

---------------------------------- 
TOPOLOGY: fat_k4_100G_OS2 
NETWORK LOAD: 52 
TIME: 0.1 
----------------------------------
 
Run Lossless RDMA experiments... 

Runing all in parallel. Check the processors running on background! 
```