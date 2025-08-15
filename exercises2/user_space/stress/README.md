


```
root@ubuntu:~/monitor# cat stress.sh 
#!/bin/bash
#stress-ng --cpu 16 --vm 2 --hdd 1 --fork 32 --timeout 1440m --metrics &
nohup stress-ng --cpu 4 --vm 2 --hdd 1 --fork 8 --timeout 1440000m --metrics >/dev/null 2>&1  &
```


```
root@ubuntu:~/monitor# nohup ./stress.sh 
nohup: ignoring input and appending output to 'nohup.out'
root@ubuntu:~/monitor# 
```