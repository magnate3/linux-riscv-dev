
# Linux  4.18.9 

```
root@xdp:~/af_xdp_demo/AF_XDP-example# uname -a
Linux xdp 4.18.9 #3 SMP Fri Feb 2 19:29:24 CST 2024 x86_64 x86_64 x86_64 GNU/Linux
root@xdp:~/af_xdp_demo/AF_XDP-example# 
AF_XDP-example# ./xdpsock -i ens3 -q 0 -t -S
ifname ens3, ifindex 2 
libbpf: elf: skipping unrecognized data section(7) .xdp_run_config
libbpf: elf: skipping unrecognized data section(8) xdp_metadata
libbpf: BTF is required, but is missing or corrupted.
xdpsock.c:xsk_configure_socket:1070: errno: 2/"No such file or directory"
```
+  clang 版本
```
root@xdp:~# clang-10 -v
clang version 10.0.0-4ubuntu1~18.04.2 
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
```


```
#define XDP_USE_SG      (1 << 4)
#define XDP_USE_NEED_WAKEUP (1 << 3)
/* Flags for xsk_umem_config flags */
#define XDP_UMEM_UNALIGNED_CHUNK_FLAG   (1 << 0)
        /* XDP_RING flags */
#define XDP_RING_NEED_WAKEUP (1 << 0)
/* Masks for unaligned chunks mode */
#define XSK_UNALIGNED_BUF_OFFSET_SHIFT 48
#define XSK_UNALIGNED_BUF_ADDR_MASK \
                ((1ULL << XSK_UNALIGNED_BUF_OFFSET_SHIFT) - 1)
#define XDP_PKT_CONTD (1 << 0)
```