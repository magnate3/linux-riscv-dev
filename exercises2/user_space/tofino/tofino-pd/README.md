
# pd


```
export SDE=/sde/bf-sde-8.9.1
export SDE_INSTALL=$SDE/install
```

```
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/tofino-pd/test# make
gcc /sde/bf-sde-8.9.1/install/include chip_cp.c -o diag_cp /usr/local/lib/libthrift-0.13.0.so /usr/lib/x86_64-linux-gnu/libpthread.so /sde/bf-sde-8.9.1/install/lib/libbfutils.so /sde/bf-sde-8.9.1/install/lib/tofinopd/diag/libpdthrift.so /sde/bf-sde-8.9.1/install/lib/tofinopd/diag/libpd.so /sde/bf-sde-8.9.1/install/lib/libdriver.so 
gcc  -I /sde/bf-sde-8.9.1/install/include chip_cp.c -o diag_cp /usr/local/lib/libthrift-0.13.0.so /usr/lib/x86_64-linux-gnu/libpthread.so /sde/bf-sde-8.9.1/install/lib/libbfutils.so /sde/bf-sde-8.9.1/install/lib/tofinopd/diag/libpdthrift.so /sde/bf-sde-8.9.1/install/lib/tofinopd/diag/libpd.so /sde/bf-sde-8.9.1/install/lib/libdriver.so 
 Finished successfully building.
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/tofino-pd/test# ls
Makefile  bf_drivers.log  chip_cp.c  commands-newtopo-tofino1.txt  commands-newtopo-tofino1.txt.bak  diag_cp  run.sh  set_bash  zlog-cfg-cur
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/tofino-pd/test# source  set_bash 
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/tofino-pd/test# ./diag_cp    
```


