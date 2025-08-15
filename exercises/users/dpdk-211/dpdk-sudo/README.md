
# sudo
```
[ubuntu test-pmd]$ export  LD_LIBRARY_PATH=/home/ubuntu/dpdk-stable-22.11.1/dpdk_dbg/lib64
[ubuntu test-pmd]$ echo $LD_LIBRARY_PATH
/home/ubuntu/dpdk-stable-22.11.1/dpdk_dbg/lib64
[ubuntu test-pmd]$ make
ln -sf helloworld-shared build/helloworld
[ubuntu test-pmd]$ sudo ./build/helloworld -c0x1
./build/helloworld: error while loading shared libraries: librte_pdump.so.23: cannot open shared object file: No such file or directory
```
root的LD_LIBRARY_PATH没有生效



## alias
alias sudo='sudo PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH="$LD_LIBRARY_PATH"'   
+ preserves HOME (which otherwise gets set to /root)   
+ preserves PATH (which otherwise gets set to a 'safe' path in suders)   
+ preserves the current value of LD_LIBRARY_PATH (which otherwise gets blanked   

```
[ubuntu test-pmd]$ alias sudo='sudo PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH="$LD_LIBRARY_PATH"'
[ubuntu test-pmd]$ sudo ./build/helloworld -c0x1
[sudo] password for ubuntu: 
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
EAL: Error - exiting with code: 1
  Cause: No cores defined for forwarding
Check the core mask argument
Port 0 is closed
```
+ only $LD_LIBRARY_PATH
```
[ubuntu test-pmd]$ alias sudo='sudo LD_LIBRARY_PATH="$LD_LIBRARY_PATH"'
[ubuntu test-pmd]$ sudo ./build/helloworld -c0x1
[sudo] password for ubuntu: 
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
EAL: Error - exiting with code: 1
  Cause: No cores defined for forwarding
Check the core mask argument
Port 0 is closed
```