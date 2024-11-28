

# 100G Benchmarking

[100G Benchmarking](https://fasterdata.es.net/host-tuning/linux/100g-tuning/100g-benchmarking/)   

[Performance Expectations for a 100G Host](https://fasterdata.es.net/performance-testing/performance-expectations-for-a-100g-host/)   




```
 ./set_irq_affinity_cpulist.sh 0-7 enp153s0f0
Discovered irqs for enp153s0f0: 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202
Assign irq 155 core_id 0
Assign irq 156 core_id 1
Assign irq 157 core_id 2
Assign irq 158 core_id 3
Assign irq 159 core_id 4
Assign irq 160 core_id 5
Assign irq 161 core_id 6
Assign irq 162 core_id 7
Assign irq 163 core_id 0
Assign irq 164 core_id 1
Assign irq 165 core_id 2
Assign irq 166 core_id 3
Assign irq 167 core_id 4
Assign irq 168 core_id 5
Assign irq 169 core_id 6
Assign irq 170 core_id 7
Assign irq 171 core_id 0
Assign irq 172 core_id 1
Assign irq 173 core_id 2
Assign irq 174 core_id 3
Assign irq 175 core_id 4
Assign irq 176 core_id 5
Assign irq 177 core_id 6
Assign irq 178 core_id 7
Assign irq 179 core_id 0
Assign irq 180 core_id 1
Assign irq 181 core_id 2
Assign irq 182 core_id 3
Assign irq 183 core_id 4
Assign irq 184 core_id 5
Assign irq 185 core_id 6
Assign irq 186 core_id 7
Assign irq 187 core_id 0
Assign irq 188 core_id 1
Assign irq 189 core_id 2
Assign irq 190 core_id 3
Assign irq 191 core_id 4
Assign irq 192 core_id 5
Assign irq 193 core_id 6
Assign irq 194 core_id 7
Assign irq 195 core_id 0
Assign irq 196 core_id 1
Assign irq 197 core_id 2
Assign irq 198 core_id 3
Assign irq 199 core_id 4
Assign irq 200 core_id 5
Assign irq 201 core_id 6
Assign irq 202 core_id 7

done.
```



```
numactl -C 8-15 iperf -P 8 -c  10.10.203.3 -p 9999
------------------------------------------------------------
Client connecting to 10.10.203.3, TCP port 9999
TCP window size:  325 KByte (default)
------------------------------------------------------------
[  1] local 10.10.203.4 port 36974 connected with 10.10.203.3 port 9999
[ ID] Interval       Transfer     Bandwidth
[  1] 0.0000-10.0082 sec  30.5 GBytes  26.2 Gbits/sec
```

# dperf


```
 apt install make    gcc  libnuma-dev -y
```
dpdk   
```
 export RTE_TARGET=x86_64-native-linuxapp-gcc
make install T=$TARGET -j16
```

dperf
```
make -j8 RTE_SDK=/root/dpdk-stable-19.11.14/ RTE_TARGET=$TARGET 
```