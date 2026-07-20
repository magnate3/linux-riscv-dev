qemu-system-x86_64 -M pc -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0" \
	-netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 -nographic \
        -drive file=virtio_blk.img ,if=none,id=drive-virtio-disk0,format=raw -device virtio-blk-pci,scsi=off,drive=drive-virtio-disk0,id=virtio-disk0,bootindex=0 \
	        -device pci-bridge,id=pci_bridge1,bus=pci.0,chassis_nr=1 \
