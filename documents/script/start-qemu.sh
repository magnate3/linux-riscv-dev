#!/bin/sh
(
BINARIES_DIR="${0%/*}/"
cd ${BINARIES_DIR}

if [ "${1}" = "serial-only" ]; then
    EXTRA_ARGS='-nographic'
else
    EXTRA_ARGS='-serial stdio'
fi

export PATH="/work/xenomai/buildroot/output/host/bin:${PATH}"
exec qemu-system-x86_64 -enable-kvm -machine q35 -cpu host -smp 4   -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0"  -net nic,model=virtio -net user  ${EXTRA_ARGS} \
 -device vfio-pci,\
	       sysfsdev=/sys/bus/mdev/devices/83b8f4f2-509f-382f-3c1e-e6bfe0fa1001  -S -gdb tcp::1234
#exec qemu-system-x86_64 -M pc -kernel bzImage -drive file=rootfs.ext2,if=virtio,format=raw -append "rootwait root=/dev/vda console=tty1 console=ttyS0"  -net nic,model=virtio -net user  ${EXTRA_ARGS}
)
