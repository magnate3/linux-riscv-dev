# hugepage-map

## 1. Enable 1GB hugepages 
To allocate four 1GB hugepages, you should append this to the kernel commandline when booting:

<code>default_hugepagesz=1G hugepagesz=1G hugepages=4</code>

To permanently add this to the kernel commandline, append it to GRUB_CMDLINE_LINUX in /etc/default/grub and then execute:

<code>$ grub2-mkconfig -o /boot/grub2/grub.cfg</code>

Reboot to make sure it worked. After rebooting, you can also use the following command to change the number of hugepages to <code>N</code>

<code>$ sysctl -w vm.nr_hugepages=N</code>

## 2. Mount 1GB hugepages on the host
<code>
$ mkdir /dev/hugepages1G
  
$ mount -t hugetlbfs -o pagesize=1G none /dev/hugepages1G
</code>

# References
https://dpdk-guide.gitlab.io/dpdk-guide/setup/hugepages.html

https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/virtualization_tuning_and_optimization_guide/sect-Virtualization_Tuning_Optimization_Guide-Memory-Huge_Pages-1GB-runtime
