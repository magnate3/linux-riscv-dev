#qemu-system-x86_64 -M pc -kernel  bzImage.nvme -drive file=rootfs.ext2,if=virtio,format=raw -append "console=ttyS0 root=/dev/vda nokaslr"   -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=foo -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic 
#########qcow2 expand
#1)qemu-img resize vm1.qcow2 +3G
# 2) and 3) do in vm
#2)growpart /dev/sda 1
#3) resize2fs /dev/sda1
#qemu-img create -f qcow2 focal-server-20G.qcow2 20G
#qemu-img info  focal-server.img 
# root pass:123456
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  \
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 qemu-system-x86_64 -enable-kvm -smp 6 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_rdma/vmlinuz-6.3.2 -append "nokaslr kgdbwait console=ttyS0 root=/dev/sda1"\
	 -hda  focal-server.img    -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=nvme-dev\
	 -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic \

	 #-S -gdb tcp::1234
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -M pc  -hda focal-server-cloudimg-amd64-disk-kvm.img    -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=nvme-dev -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic
