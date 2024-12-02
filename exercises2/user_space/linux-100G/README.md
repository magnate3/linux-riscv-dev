

# 100G Benchmarking

[100G Benchmarking](https://fasterdata.es.net/host-tuning/linux/100g-tuning/100g-benchmarking/)   

[Performance Expectations for a 100G Host](https://fasterdata.es.net/performance-testing/performance-expectations-for-a-100g-host/)   



./set_irq_affinity_cpulist.sh 0-7 enp153s0f0    
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
./set_irq_affinity_cpulist.sh  0,2,4,6,8,10,12,14 enp23s0f0     
```
./set_irq_affinity_cpulist.sh  0,2,4,6,8,10,12,14 enp23s0f0
Discovered irqs for enp23s0f0: 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
Assign irq 154 core_id 0
Assign irq 155 core_id 2
Assign irq 156 core_id 4
Assign irq 157 core_id 6
Assign irq 158 core_id 8
Assign irq 159 core_id 10
Assign irq 160 core_id 12
Assign irq 161 core_id 14
Assign irq 162 core_id 0
Assign irq 163 core_id 2
Assign irq 164 core_id 4
Assign irq 165 core_id 6
Assign irq 166 core_id 8
Assign irq 167 core_id 10
Assign irq 168 core_id 12
Assign irq 169 core_id 14
Assign irq 170 core_id 0
Assign irq 171 core_id 2
Assign irq 172 core_id 4
Assign irq 173 core_id 6
Assign irq 174 core_id 8
Assign irq 175 core_id 10
Assign irq 176 core_id 12
Assign irq 177 core_id 14
Assign irq 178 core_id 0
Assign irq 179 core_id 2
Assign irq 180 core_id 4
Assign irq 181 core_id 6
Assign irq 182 core_id 8
Assign irq 183 core_id 10
Assign irq 184 core_id 12
Assign irq 185 core_id 14
Assign irq 186 core_id 0
Assign irq 187 core_id 2
Assign irq 188 core_id 4
Assign irq 189 core_id 6
Assign irq 190 core_id 8
Assign irq 191 core_id 10
Assign irq 192 core_id 12
Assign irq 193 core_id 14
Assign irq 194 core_id 0
Assign irq 195 core_id 2
Assign irq 196 core_id 4
Assign irq 197 core_id 6
Assign irq 198 core_id 8
Assign irq 199 core_id 10
Assign irq 200 core_id 12
Assign irq 201 core_id 14

done.
```

+ ./show_irq_affinity.sh  enp23s0f0    
```
./show_irq_affinity.sh  enp23s0f0
155: 2
156: 40
157: 4
158: 16
159: 0
160: 20
161: 46
162: 44
163: 2
164: 22
165: 42
166: 14
167: 10
168: 10
169: 12
170: 36
171: 18
172: 38
173: 32
174: 34
175: 16
176: 12
177: 14
178: 10
179: 14
180: 12
181: 4
182: 0
183: 26
184: 28
185: 24
186: 46
187: 8
188: 16
189: 6
190: 40
191: 44
192: 40
193: 20
194: 42
195: 36
196: 18
197: 28
198: 38
199: 30
200: 32
201: 42
202: 34
```

+ set_irq_affinity_bynode.sh    

```bash
   ./set_irq_affinity_bynode.sh `\cat /sys/class/net/eth0/device/numa_node` eth0
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
iperf uses base 2 for M and K, meaning that K = 1024 and M = 1024*1024.

```
numactl -C 24,26,27,28,30,32,34,36 iperf -s   -p 9999
```

# dperf

```
0000:3d:00.0 'Ethernet Controller E810-C for QSFP 1592' if=enp61s0f0 drv=ice unused=vfio-pci,uio_pci_generic 
```

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

