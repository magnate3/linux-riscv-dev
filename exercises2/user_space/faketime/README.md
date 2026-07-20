

```
git clone https://github.com/wolfcw/libfaketime.git
cd libfaketime
make
make install # not need
```

```
root@ubuntux86:# ls /work/tna/libfaketime/src/libfaketime.so.1
/work/tna/libfaketime/src/libfaketime.so.1
root@ubuntux86:# chmod 777  /work/tna/libfaketime/src/libfaketime.so.1
```

```
ubuntu@ubuntux86:p4i$ export LD_PRELOAD=/work/tna/libfaketime/src/libfaketime.so.1
ubuntu@ubuntux86:p4i$ export FAKETIME="@2021-11-16 15:30:00"
ubuntu@ubuntux86:p4i$ date
2021年 11月 16日 星期二 15:30:00 CST
```