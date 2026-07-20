
```
root@ubuntux86:#  cat /dev/net/tun
cat: /dev/net/tun: File descriptor in bad state
root@ubuntux86:# ls -l /dev/net/tun
crw-rw-rw- 1 root root 10, 200 12月  8 09:21 /dev/net/tun
root@ubuntux86:# ls -l /dev/net/tun
crw-rw-rw- 1 root root 10, 200 12月  8 09:21 /dev/net/tun
root@ubuntux86:# 
root@ubuntux86:# ls -l /dev/net/tun
crw-rw-rw- 1 root root 10, 200 12月  8 09:21 /dev/net/tun
root@ubuntux86:# lsmod | grep tun
root@ubuntux86:# modprobe tun
root@ubuntux86:# ./src/bin/netsim  tap1 tap2
```

```
./src/bin/netsim  tap1 tap2 -v
```