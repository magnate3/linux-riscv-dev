

```
root@ubuntux86:# gcc malloc_test.c  -o malloc_test
root@ubuntux86:# ldd malloc_test
        linux-vdso.so.1 (0x00007ffe72f13000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fa0bd30f000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fa0bd526000)
```


```
root@ubuntux86:# ./malloc_test 
pid :3038

```

bpftrace   
```
root@ubuntux86:# bpftrace memory.bt
Attaching 3 probes...
call malloc size: 8
alloc memory 0x563787494ac0
call malloc size: 8
alloc memory 0x563787494ae0
call malloc size: 8
alloc memory 0x563787494b00
call malloc size: 8
alloc memory 0x563787494b20
call malloc size: 8
alloc memory 0x563787494b40
call malloc size: 8
alloc memory 0x563787494b60
call malloc size: 8
alloc memory 0x563787494b80
call malloc size: 8
alloc memory 0x563787494ba0
call malloc size: 8
alloc memory 0x563787494bc0
call malloc size: 8
alloc memory 0x563787494be0
```