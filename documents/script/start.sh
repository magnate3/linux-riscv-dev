qemu-system-x86_64 -M pc -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0"  -net nic,model=virtio -net user  -S -gdb tcp::1234
