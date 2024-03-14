


```
[ubuntu tep_termination]$ mount | grep huge
cgroup on /sys/fs/cgroup/hugetlb type cgroup (rw,nosuid,nodev,noexec,relatime,seclabel,hugetlb)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,seclabel)
[ubuntu tep_termination]$ sudo ./build/app/tep_termination  -c f -n 4 --huge-dir /dev/hugepages --  -p 0x1 --dev-basename tep-termination --nb-devices 4   --udp-port 4789 --filter-type 1


```
[ubuntu tep_termination]$ rm tep-termination 
rm: remove write-protected socket ‘tep-termination’? y
[ubuntu tep_termination]$ sudo ./build/app/tep_termination  -c f -n 4 --huge-dir /dev/hugepages --  -p 0x1 --dev-basename tep-termination --nb-devices 4   --udp-port 4789 --filter-type 1
EAL: Detected 72 lcore(s)
```

# references

[vxlan_sample_test_plan](https://oss.iol.unh.edu/dpdk/dts/-/blob/for-next-v3-trex-single-core-perf/test_plans/vxlan_sample_test_plan.rst)