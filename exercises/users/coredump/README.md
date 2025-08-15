root@ubuntux86:# echo 'ulimit -c unlimited' >> ~/.bashrc    
root@ubuntux86:# echo "/work/coredump/core.%e.%p"> /proc/sys/kernel/core_pattern    