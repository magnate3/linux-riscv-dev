#qemu-system-x86_64 -enable-kvm -smp 4 -m 8G -machine q35 -hda focal-server-cloudimg-amd64-disk-kvm.img -nographic \
#qemu-system-x86_64 -M pc -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0" \
#	-netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 -nographic  -S -gdb tcp::1234
# good
#qemu-system-x86_64 -M pc  -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0 nokaslr" \
#	-netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 -nographic
       #	-S -gdb tcp::1234
#qemu-system-x86_64 -M pc -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0 raid=noautodetect" \
#	-netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 -nographic 
#############good
qemu-system-x86_64 -enable-kvm -smp 4 -m 8G -machine pc-q35-4.2    -hda focal-server-cloudimg-amd64-disk-kvm.img -nographic \
	-netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01
#qemu-system-x86_64 -enable-kvm -smp 4 -m 8G -machine pc-q35-4.2 -hda focal-server-cloudimg-amd64-disk-kvm.img -nographic \
#  -chardev socket,id=chardev0,path=vhost-user.sock \
#         -netdev vhost-user,chardev=chardev0,id=netdev0 \
#	        -device virtio-net-pci,netdev=netdev0
