#!/bin/bash
:<<!
/opt/qemu-cxl/bin/qemu-system-x86_64 -drive file=CXL-Test.qcow2,format=qcow2,index=0,media=disk,id=hd \
	-m 4G,slots=8,maxmem=8G \
	-smp 4 \
	-machine type=q35,accel=kvm,nvdimm=on,cxl=on \
	-enable-kvm \
	-nographic \
	-net nic \
	-net user,hostfwd=tcp::2222-:22
!

#/opt/qemu-cxl/bin/qemu-system-x86_64 -drive file=CXL-Test.qcow2,format=qcow2,index=0,media=disk,id=hd \
#	-m 4G,slots=8,maxmem=8G \
#	-smp 4 \
#	-machine type=q35,accel=kvm,nvdimm=on,cxl=on \
#	-enable-kvm \
#        -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 \
#	-nographic \

#:<<!
/opt/qemu-jic23/bin/qemu-system-x86_64 -drive file=CXL-Test.qcow2,format=qcow2,index=0,media=disk,id=hd \
	-m 4G,slots=8,maxmem=8G \
	-smp 4 \
	-machine type=q35,accel=kvm,nvdimm=on,cxl=on \
	-enable-kvm \
        -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 \
	-nographic \
	-m 2G,slots=8,maxmem=6G \
	  -smp cpus=4,cores=2,sockets=2 \
	-object memory-backend-ram,size=1G,id=mem0 \
	  -numa node,nodeid=0,cpus=0-1,memdev=mem0 \
	    -object memory-backend-ram,size=1G,id=mem1 \
	      -numa node,nodeid=1,cpus=2-3,memdev=mem1 \
	        -object memory-backend-ram,id=vmem0,share=on,size=256M \
		  -object memory-backend-ram,id=vmem1,share=on,size=256M \
		    -device pxb-cxl,numa_node=0,bus_nr=12,bus=pcie.0,id=cxl.1 \
		      -device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=2 \
		        -device cxl-type3,bus=root_port13,volatile-memdev=vmem0,id=cxl-vmem0 \
			  -device cxl-rp,port=0,bus=cxl.1,id=root_port14,chassis=1,slot=2 \
			    -device cxl-type3,bus=root_port14,volatile-memdev=vmem1,id=cxl-vmem1 \
			      -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G

#!
# this ok
#/opt/qemu-jic23/bin/qemu-system-x86_64 -drive file=CXL-Test.qcow2,format=qcow2,index=0,media=disk,id=hd \
#	-m 4G,slots=8,maxmem=8G \
#	-smp 4 \
#	-machine type=q35,accel=kvm,nvdimm=on,cxl=on \
#	-enable-kvm \
#        -netdev tap,id=tap0,ifname=tap0,script=no,downscript=no,vhost=on  -device virtio-net-pci,netdev=tap0,mac=52:55:00:d1:55:01 \
#	-nographic \
#	-object memory-backend-ram,size=4G,id=mem0 \
#	-numa node,nodeid=0,cpus=0-3,memdev=mem0 \
#	-object memory-backend-file,id=pmem0,share=on,mem-path=/tmp/cxltest.raw,size=256M \
#	-object memory-backend-file,id=cxl-lsa0,share=on,mem-path=/tmp/lsa.raw,size=256M \
#	-device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.1 \
#	-device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=2 \
#	-device cxl-type3,bus=root_port13,persistent-memdev=pmem0,lsa=cxl-lsa0,id=cxl-pmem0 \
#	-M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G
