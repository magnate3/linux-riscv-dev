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
