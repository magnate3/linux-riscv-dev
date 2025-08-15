#qemu-system-x86_64 -M pc -kernel  bzImage.nvme -drive file=rootfs.ext2,if=virtio,format=raw -append "console=ttyS0 root=/dev/vda nokaslr"   -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=foo -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic 
#########qcow2 expand
#1)qemu-img resize vm1.qcow2 +3G
# 2) and 3) do in vm
#2)growpart /dev/sda 1
#3) resize2fs /dev/sda1
#qemu-img create -f qcow2 focal-server-20G.qcow2 20G
#qemu-img info  focal-server.img 
# root pass:123456
# root change pass virt-sysprep --root-password password:123456 -a bionic-server-cloudimg-arm64.img
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  \
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_nokaslr/vmlinuz-6.3.2 -append "console=ttyS0 root=/dev/sda1 nokaslr"\
 qemu-system-x86_64    -enable-kvm -smp 6 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_rdma/vmlinuz-6.3.2 -append "nokaslr kgdbwait console=ttyS0 root=/dev/sda1"\
	 -hda  focal-server.img \
	    -device pcie-root-port,id=root_port1,chassis=1,slot=1 \
	        -device x3130-upstream,id=upstream1,bus=root_port1 \
	 -device xio3130-downstream,id=downstream1,bus=upstream1,chassis=9 \
	 -device xio3130-downstream,id=downstream2,bus=upstream1,chassis=10 \
	 -drive file=nvme.img,if=none,id=nvme0 \
	 -device nvme,drive=nvme0,serial=d00d0001,cmb_size_mb=1024,bus=downstream1 \
	 -drive file=nvme2.img,if=none,id=nvme1 \
	 -device nvme,drive=nvme1,serial=d00d0002,cmb_size_mb=1024,bus=downstream2 \
	 -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic 


'''
 qemu-system-x86_64 -enable-kvm -smp 6 -m 8G  -cpu host  -kernel  /work/linux-6.3.2/k_rdma/vmlinuz-6.3.2 -append "nokaslr kgdbwait console=ttyS0 root=/dev/sda1"\
       -hda  focal-server.img -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=nvme-dev,,cmb_size_mb=1024\
       -drive file=nvme2.img,if=none,format=raw,id=drv2 -device nvme,drive=drv2,serial=nvme-dev2,cmb_size_mb=1024\
       -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic \

       #-S -gdb tcp::1234
'''
 #qemu-system-x86_64 -enable-kvm -smp 4 -m 8G  -M pc  -hda focal-server-cloudimg-amd64-disk-kvm.img    -drive file=nvme.img,if=none,format=raw,id=drv0 -device nvme,drive=drv0,serial=nvme-dev -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01  -nographic
