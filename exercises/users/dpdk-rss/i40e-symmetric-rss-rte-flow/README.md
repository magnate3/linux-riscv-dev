
# run

```
[ubuntu i40e-symmetric-rss-rte-flow]$ sudo ./build/test_rss -c0x1 -- port0
[sudo] password for liangjun: 
TELEMETRY: No legacy callbacks, legacy socket not created
name is 0000:1a:00.1 
Error (err=-19) when getting port id of port0 Is device enabled?
device_configure(): unable to configure the device, -19
[ubuntu i40e-symmetric-rss-rte-flow]$ sudo ./build/test_rss -c0x1 -- 0000:1a:00.1 
TELEMETRY: No legacy callbacks, legacy socket not created
name is 0000:1a:00.1 
Creating a packet mbuf pool pktmbuf_pool_p0_q0 of size 262143, cache size 500, mbuf size 2176
Creating Q 0 of P 0 using desc RX: 4096 TX: 4096 RX htresh: 0 RX pthresh 0 wtresh 0 free_tresh 0 drop_en 0 Offloads 14
i40e_hash_parse_queues(): RSS key is ignored when queues specified
RULE1 created
RULE2 created
RULE3 created
RULE4 created

Starting all cores ... [Ctrl+C to quit]


Received 1 packet/s on lcore 0
Received 1 packet/s on lcore 0
Received 1 packet/s on lcore 0
Received 1 packet/s on lcore 0
^CLcoreid 0 total rx count: 4
Thread 0 finished
```


# Setting symmetric RSS on I40E with rte_flow
 
After too much of "pain and tears" I was able to configure receive side scaling with symmetric toeplitz function on I40E PMD driver. To save you from the trouble I built a little example to show you how it is set.

The main thing was that it can only be set after the port has started.

How to set full RSS support with testpmd:
```
flow create 0 ingress pattern end actions rss types end queues 0 end / end
flow create 0 ingress pattern eth / ipv4 / end actions rss func symmetric_toeplitz types ipv4-other end queues end / end
flow create 0 ingress pattern eth / ipv4 / tcp / end actions rss func symmetric_toeplitz types ipv4-tcp end queues end / end
flow create 0 ingress pattern eth / ipv4 / udp / end actions rss func symmetric_toeplitz types ipv4-udp end queues end / end
flow create 0 ingress pattern eth / ipv4 / sctp / end actions rss func symmetric_toeplitz types ipv4-sctp end queues end / end
flow create 0 ingress pattern eth / ipv6 / end actions rss func symmetric_toeplitz types ipv6-other end queues end / end
flow create 0 ingress pattern eth / ipv6 / tcp / end actions rss func symmetric_toeplitz types ipv6-tcp end queues end / end
flow create 0 ingress pattern eth / ipv6 / udp / end actions rss func symmetric_toeplitz types ipv6-udp end queues end / end
flow create 0 ingress pattern eth / ipv6 / sctp / end actions rss func symmetric_toeplitz types ipv6-sctp end queues end / end
```

In the current example, only the first 4 lines are set but the others can be set accordingly to the pattern.

Hope it helps.
