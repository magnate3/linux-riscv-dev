

# insmod  map_test.ko

```
[root@centos7 bitmap]# dmesg | tail -n 10
[ 4667.268159] test begin >>>>>>>>>>>>>>>>>>> 
[ 4667.272328]  BITS_PER_LONG 64
[ 4667.275298] Bitmap(0):   0xffffffffffffffff
[ 4667.279461] Bitmap(1):   0xfffffffffffffffe
[ 4667.283625] Bitmap(2):   0xfffffffffffffffc
[ 4667.287794] Bitmap(3):   0xfffffffffffffff8
[ 4667.291957] before set 0x0, 0x0
[ 4667.295090] after set 0xfffffffffffffff0, 0xf
[ 4667.299426] after set bits all  0xfff
[ 4667.303072] after clear set bits all  0x0
```