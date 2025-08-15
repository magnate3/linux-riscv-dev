#/bin/bash
#qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu IvyBridge  -hda ubuntu-14.04-server-cloudimg-amd64-disk1.img \
#      	-netdev tap,id=tap1,ifname=tap1,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap1,mac=52:55:00:d1:55:02 \
#	-netdev tap,id=tap0,ifname=tap0,script=no,vhost=on -device virtio-net-pci,netdev=tap0,ioeventfd=on\
#        	  -nographic

qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu IvyBridge -kernel  /work/linux-6.3.2/k_rdma_gdb/vmlinuz-6.3.2 -append "nokaslr kgdbwait console=ttyS0 root=/dev/sda1" \
        -hda  focal-server-ovs.img    \
         -netdev tap,id=tap0,ifname=tap0,script=no,vhost=on -device virtio-net-pci,netdev=tap0,ioeventfd=on\
	 -netdev tap,id=tap1,ifname=tap1,script=no,downscript=no,vhost=off  -device e1000,id=e1,netdev=tap1,mac=52:55:00:d1:55:02\
        -nographic

#qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu IvyBridge  -hda ubuntu-14.04-server-cloudimg-amd64-disk1.img \
#	-net nic,model=e1000,macaddr=DE:AD:1E:00:00:01 -net tap,ifname=tap0,script=no,downscript=no \
#	-net nic,model=e1000,macaddr=DE:AD:1E:00:00:02 -net tap,ifname=tap1,script=no,downscript=no \
#        -nographic
	#-net nic,model=e1000,macaddr=DE:AD:1E:00:00:01 \
        #-net tap,ifname=tap0,script=no,downscript=no \
        #-net nic,model=e1000,macaddr=DE:AD:1E:00:00:02 \
        #-net tap,ifname=tap1,script=no,downscript=no \
        # -nographic
