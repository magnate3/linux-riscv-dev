cat /proc/interrupts |grep -i nvme[0-3]q0|awk '{print $1,$(NF-2),$(NF-1),$NF}'
201: PCI-MSI 118489088-edge nvme1q0
203: PCI-MSI 119013376-edge nvme2q0
205: PCI-MSI 119537664-edge nvme3q0
399: PCI-MSI 117964800-edge nvme0q0

cat /proc/interrupts |grep nvme |wc -l
516




```
[root@centos7 linux-6.3]# cat /proc/interrupts |grep -i  arm-smmu-v3-priq |awk '{print $1,$(NF-2),$(NF-1),$NF}'
21: 100354 Edge arm-smmu-v3-priq
24: 102402 Edge arm-smmu-v3-priq
27: 104450 Edge arm-smmu-v3-priq
30: 106498 Edge arm-smmu-v3-priq
33: 108546 Edge arm-smmu-v3-priq
36: 110594 Edge arm-smmu-v3-priq
39: 112642 Edge arm-smmu-v3-priq
42: 114690 Edge arm-smmu-v3-priq
[root@centos7 linux-6.3]# 
```