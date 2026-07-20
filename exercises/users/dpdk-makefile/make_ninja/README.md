
# ninja 转make出现 undefined reference to
`${DPDK_LIB}/lib -ldpdk`     
在app编译时加上LDFLAGS +=  -L ${DPDK_LIB}/lib -ldpdk -lpthread -lnuma -lrt -lm -ldl   
```
../build/libstack.so: undefined reference to `rte_hash_add_key_data'
../build/libstack.so: undefined reference to `rte_mempool_ops_table'
../build/libstack.so: undefined reference to `rte_eth_macaddr_get'
../build/libstack.so: undefined reference to `rte_timer_subsystem_init'
../build/libstack.so: undefined reference to `rte_exit'
../build/libstack.so: undefined reference to `rte_eth_dev_configure'
../build/libstack.so: undefined reference to `rte_mempool_check_cookies'
../build/libstack.so: undefined reference to `rte_free'
../build/libstack.so: undefined reference to `rte_log'
../build/libstack.so: undefined reference to `rte_eal_init'
../build/libstack.so: undefined reference to `per_lcore__lcore_id'
../build/libstack.so: undefined reference to `rte_timer_init'
../build/libstack.so: undefined reference to `rte_eth_devices'
../build/libstack.so: undefined reference to `rte_hash_del_key'
../build/libstack.so: undefined reference to `rte_get_next_lcore'
../build/libstack.so: undefined reference to `rte_kni_tx_burst'
../build/libstack.so: undefined reference to `rte_get_tsc_cycles'
../build/libstack.so: undefined reference to `rte_hash_iterate'
../build/libstack.so: undefined reference to `rte_eth_dev_start'
../build/libstack.so: undefined reference to `rte_socket_id'
../build/libstack.so: undefined reference to `rte_eth_tx_queue_setup'
../build/libstack.so: undefined reference to `rte_eth_dev_count_avail'
../build/libstack.so: undefined reference to `__rte_panic'
../build/libstack.so: undefined reference to `rte_eth_rx_queue_setup'
../build/libstack.so: undefined reference to `rte_eal_cleanup'
../build/libstack.so: undefined reference to `rte_get_timer_hz'
../build/libstack.so: undefined reference to `rte_pktmbuf_pool_create'
../build/libstack.so: undefined reference to `rte_eth_promiscuous_enable'
../build/libstack.so: undefined reference to `rte_eth_dev_socket_id'
```

# undefined reference to `rte_get_tsc_cycles'
```
../build/libstack.so: undefined reference to `rte_get_tsc_cycles'
../build/libstack.so: undefined reference to `rte_get_timer_hz'
collect2: error: ld returned 1 exit libstack.so
```
查看使用了rte_get_tsc_cycles的文件    

```
include/arp_table.h:100:        arp_entry->timeout = rte_get_tsc_cycles() / rte_get_tsc_hz() + ARP_ENTRY_TIMEOUT;
include/arp_table.h:108:        arp_entry->timeout = rte_get_tsc_cycles() / rte_get_tsc_hz() + ARP_ENTRY_TIMEOUT;
src/arp.c:65:    uint64_t cur_tsc = rte_get_tsc_cycles();
src/netdev.c:77:    uint64_t cur_tsc = rte_get_tsc_cycles();
```
在上述文件中加上#include <rte_cycles.h>   

#    error while loading shared libraries: libstack.so
```
[root@centos7 app]# ./stack  -h
./stack: error while loading shared libraries: libstack.so: cannot open shared object file: No such file or directory
[root@centos7 app]# make
make[1]: Entering directory '/root/dpdk-19.11/examples/stack'
make[1]: Nothing to be done for 'all'.
make[1]: Leaving directory '/root/dpdk-19.11/examples/stack'
```
执行：

```
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../build
```