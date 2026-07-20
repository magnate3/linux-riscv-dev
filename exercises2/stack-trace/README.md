

# xdp   

```
ip netns exec ns2 ./xdpsock_user -f  -i host2   -q 0 -n 5 -S
```

查看日志

```
cat  /sys/kernel/debug/tracing/trace
```

```
   simple_switch-30625   [004] d.s1 145567.688470: bpf_trace_printk: ip next proto: 6 pass 

   simple_switch-30625   [004] d.s1 145567.688473: bpf_trace_printk: ip next proto: 1536 

   simple_switch-30625   [004] d.s1 145567.688562: bpf_trace_printk: ip next proto: 6 pass 

   simple_switch-30625   [004] d.s1 145567.688565: bpf_trace_printk: ip next proto: 1536 
```

## xdp make
os    
```
root@ubuntux86:# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:# 
```

```
root@ubuntux86:# git clone https://github.com/bhaskar792/XDP-nat64.git
root@ubuntux86:# cd XDP-nat64/
root@ubuntux86:# ls
common  headers  libbpf  loader  nat64  README.md
root@ubuntux86:# cd nat64/
root@ubuntux86:# ls
Makefile  ret_pkt.py  send_pkt.py  testbed.py  xdp_prog_kern.c  xdp_prog_kern-comments.c  xdp_prog_user.c
root@ubuntux86:# make
make[1]: Entering directory '/work/xdp-tutorial/XDP-nat64/libbpf/src'
  MKDIR    staticobjs
  CC       bpf.o
  CC       btf.o
  CC       libbpf.o
  CC       libbpf_errno.o
  CC       netlink.o
  CC       nlattr.o
  CC       str_error.o
  CC       libbpf_probes.o
  CC       bpf_prog_linfo.o
  CC       xsk.o
  CC       btf_dump.o
  CC       hashmap.o
  CC       ringbuf.o
  AR       libbpf.a
  MKDIR    sharedobjs
  CC       bpf.o
  CC       btf.o
  CC       libbpf.o
  CC       libbpf_errno.o
  CC       netlink.o
  CC       nlattr.o
  CC       str_error.o
  CC       libbpf_probes.o
  CC       bpf_prog_linfo.o
  CC       xsk.o
  CC       btf_dump.o
  CC       hashmap.o
  CC       ringbuf.o
  CC       libbpf.so.0.3.0
```