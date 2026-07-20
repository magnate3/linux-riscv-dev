

[RoCE/RDMA Tools](https://enterprise-support.nvidia.com/s/article/roce-rdma-tools)    
#  lldptool/dcbtool
Manages the LLDP setting and status of lldpad (IEEE/CEE).   

Handles lldptool-pfc (for lossless network) (here)    
```
Enable PFC for priorities 1, 2, and 4 on eth2

lldptool -T -i eth2 -V PFC enabled=1,2,4
Display priorities enabled for PFC on eth2
lldptool -t -i eth2 -V PFC -c enabled
Display last transmitted PFC TLV on eth2
lldptool -t -i eth2 -V PFC
```