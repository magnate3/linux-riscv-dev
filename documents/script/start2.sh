qemu-system-x86_64  -enable-kvm \
	  -smp 4 \
	    -m 8G \
	      -machine q35 \
	      -hda focal-server-cloudimg-amd64-disk-kvm.img \
		    -nographic 
	      #-device vfio-pci,sysfsdev=/sys/bus/mdev/devices/83b8f4f2-509f-382f-3c1e-e6bfe0fa1001
