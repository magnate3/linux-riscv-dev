insmod rmem_rdma.ko servers=10.10.49.89:18516:$((253572));
sleep 1
mkfs.ext4 `ls /dev/rmem_rdma*` 
sleep 1
mount `ls /dev/rmem_rdma*` ~/temp
