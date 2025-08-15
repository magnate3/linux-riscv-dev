
# sever

```
sudo ./build/testpmd -c0x3 -n 4 --log-level=8  -- -i  --rxq=16 --txq=16  --forward-mode=icmpecho
EAL: Detected CPU lcores: 72
EAL: Detected NUMA nodes: 2
EAL: Detected shared linkage of DPDK
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'
EAL: VFIO support initialized
EAL: Using IOMMU type 1 (Type 1)
EAL: Ignore mapping IO port bar(1)
EAL: Ignore mapping IO port bar(4)
EAL: Probe PCI driver: net_i40e (8086:37d0) device: 0000:1a:00.1 (socket 0)
TELEMETRY: No legacy callbacks, legacy socket not created
Interactive-mode selected
Set icmpecho packet forwarding mode
testpmd: create a new mbuf pool <mb_pool_0>: n=155456, size=2176, socket=0
testpmd: preferred mempool ops selected: ring_mp_mc
Configuring Port 0 (socket 0)
Port 0: F4:1D:6B:F7:BF:96
Checking link statuses...
testpmd> start
icmpecho packet forwarding - ports=1 - cores=1 - streams=16 - NUMA support enabled, MP allocation mode: native
```
# client
```
[lubuntu tcpreplay]$ sudo  arp -s 10.11.11.65  F4:1D:6B:F7:BF:96
[lubuntu tcpreplay]$ ping 10.11.11.65
PING 10.11.11.65 (10.11.11.65) 56(84) bytes of data.
64 bytes from 10.11.11.65: icmp_seq=13 ttl=64 time=0.041 ms
64 bytes from 10.11.11.65: icmp_seq=14 ttl=64 time=0.045 ms
64 bytes from 10.11.11.65: icmp_seq=15 ttl=64 time=0.031 ms
64 bytes from 10.11.11.65: icmp_seq=16 ttl=64 time=0.028 ms
64 bytes from 10.11.11.65: icmp_seq=17 ttl=64 time=0.030 ms
64 bytes from 10.11.11.65: icmp_seq=18 ttl=64 time=0.037 ms
64 bytes from 10.11.11.65: icmp_seq=19 ttl=64 time=0.037 ms
```