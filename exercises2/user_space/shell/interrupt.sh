cat /proc/interrupts |grep -i nvme[0-3]q0|awk '{print $1,$(NF-2),$(NF-1),$NF}'
201: PCI-MSI 118489088-edge nvme1q0
203: PCI-MSI 119013376-edge nvme2q0
205: PCI-MSI 119537664-edge nvme3q0
399: PCI-MSI 117964800-edge nvme0q0

cat /proc/interrupts |grep nvme |wc -l
516