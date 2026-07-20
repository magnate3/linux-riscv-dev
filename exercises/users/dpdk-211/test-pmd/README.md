
# define
+ RTE_NET_I40E
```
get_fdir_info(uint16_t port_id, struct rte_eth_fdir_info *fdir_info,
                    struct rte_eth_fdir_stats *fdir_stat)
{
        int ret = -ENOTSUP;

#ifdef RTE_NET_I40E
        if (ret == -ENOTSUP) {
                ret = rte_pmd_i40e_get_fdir_info(port_id, fdir_info);
                if (!ret)
                        ret = rte_pmd_i40e_get_fdir_stats(port_id, fdir_stat);
        }
#endif
```

# rte_os_shim.h: No such file or directory
```
testpmd.h:16:25: fatal error: rte_os_shim.h: No such file or directory
 #include <rte_os_shim.h>
```
注释#include <rte_os_shim.h>

# 命令行

```
sudo ./build/testpmd -c0x3 -n 4 --log-level=8  -- -i
```

+ -d LIB.so|DIR   
```
  -d LIB.so|DIR       Add a driver or driver directory
                      (can be used multiple times)
```
```
/dpdk/build/app/testpmd -d /usr/lib64/librte_pmd_i40e.so -l 0,1,2 --socket-mem 1024,0 -n 4 --proc-type auto --file-prefix pg $pci_devs_opt -- --nb-cores=2 --nb-ports=2 --portmask=3 --interactive --auto-start
sudo ./testpmd -l 0-5 -- -i --nb-cores=1 --forward-mode=rxonly --auto-start --portmask=0x2
```

```
testpmd>  show port info 0
./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16
testpmd> set fwd rxonly
testpmd> set verbose 1
testpmd> start

```
+ --forward-mode=rxonly
```
testpmd -w 02:00.0 -- -i --rxq=4 --txq=4 --forward-mode=rxonly
testpmd> port config all rss all
testpmd> set verbose 1
testpmd> start
```

# flow    

```
testpmd> flow create 0 ingress pattern eth / ipv4 / end actions rss / end
Port 0: link state change event

port_flow_complain(): Caught PMD error type 15 (action configuration): RSS Queues not supported when pattern specified: Invalid argument
testpmd> flow dump 0 rule 0
Failed to dump to flow 0
testpmd> flow dump 0 all
port_flow_complain(): Caught PMD error type 1 (cause unspecified): Function not implemented: Function not implemented
Failed to dump flow: Function not implemented
testpmd>
```

##  test one

```
 sudo ./build/testpmd -c0x3 -n 4 --log-level=8  -- -i  --rxq=16 --txq=16
testpmd> set fwd rxonly
Set rxonly packet forwarding mode
testpmd> set verbose 1
Change verbose level from 0 to 1
testpmd> start
testpmd> flow validate 0 ingress pattern eth type is 0x0806 / end actions queue index 1 / end
Flow rule validated
testpmd> flow create 1 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 / udp src is 22 dst is 23 / raw relative is 1 offset is 2 pattern is fhds / end actions queue index 3 / end
```