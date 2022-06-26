# insmod map_test.ko 
```
[root@centos7 bitmap]# dmesg | tail -n 16
[683642.042893] bit 0 val 1 
[683642.045506] bit 1 val 0 
[683642.048128] bit 2 val 1 
[683642.050736] bit 3 val 0 
[683642.053344] bit 4 val 1 
[683642.055952] bit 5 val 0 
[683642.058567] bit 6 val 1 
[683642.061175] bit 7 val 0 
[683642.063783] bit 0 val 0 
[683642.066395] bit 1 val 0 
[683642.069004] bit 2 val 0 
[683642.071611] bit 3 val 0 
[683642.074219] bit 4 val 0 
[683642.076831] bit 5 val 0 
[683642.079439] bit 6 val 0 
[683642.082047] bit 7 val 0 
```
# insmod map_test2.ko 
```
[root@centos7 bitmap]# dmesg | tail -n 16
[688042.267477] **bit 0 val 1 
[688042.270263] **bit 1 val 0 
[688042.273054] **bit 2 val 1 
[688042.275836] **bit 3 val 0 
[688042.278617] **bit 4 val 1 
[688042.281403] **bit 5 val 0 
[688042.284184] **bit 6 val 1 
[688042.286965] **bit 7 val 0 
[688042.289746] ** bit 0 val 0 
[688042.292619] ** bit 1 val 0 
[688042.295485] ** bit 2 val 0 
[688042.298352] ** bit 3 val 0 
[688042.301224] ** bit 4 val 0 
[688042.304091] ** bit 5 val 0 
[688042.306957] ** bit 6 val 0 
[688042.309823] ** bit 7 val 0 
```