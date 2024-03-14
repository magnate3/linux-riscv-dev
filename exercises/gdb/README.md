
# gdb

gdb 先查看断点号，然后删除断点
i b --查看断点号
d 断点号--删除断点



```
(gdb) i b
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0xffffffff8029fbfa in pci_ecam_map_bus at drivers/pci/ecam.c:170
        breakpoint already hit 1 time
2       breakpoint     keep y   0xffffffff80282234 in pci_generic_config_read at drivers/pci/access.c:82
        breakpoint already hit 1 time
(gdb) d 1
(gdb) d 2
(gdb) c
Continuing.
```

# gdb kernel

```
 riscv64-unknown-elf-gdb   ~/riscv_debug/linux-5.14/vmlinux
```

