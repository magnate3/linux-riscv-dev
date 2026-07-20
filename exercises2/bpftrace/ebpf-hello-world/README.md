
#  聊聊对 BPF 程序至关重要的 vmlinux.h 文件

[聊聊对 BPF 程序至关重要的 vmlinux.h 文件](https://www.ebpf.top/post/intro_vmlinux_h/)

# libbpf


```
git clone https://github.com/libbpf/libbpf && cd libbpf/src/
make BUILD_STATIC_ONLY=1 OBJDIR=../build/libbpf DESTDIR=../build INCLUDEDIR=LIBDIR=UAPIDIR=install
```


# make helloworld

+ 1  生成vmlinux.h   
```
root@ubuntux86:# pwd
/work/kernel_learn/bpftrace/ebpf-hello-world/src
root@ubuntux86:# bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
root@ubuntux86:# 
```

+ 2  直接make


```
root@ubuntux86:# pwd
/work/kernel_learn/bpftrace/ebpf-hello-world/src
root@ubuntux86:# make
uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
mkdir -p build
bpftool btf dump file /sys/kernel/btf/vmlinux format c > build/vmlinux.h
clang -g -O2 -Wall -Wextra -target bpf -D__TARGET_ARCH_x86_64 -I../libbpf/src -c  hello.bpf.c -o build/hello.bpf.o
hello.bpf.c:5:78: warning: unused parameter 'ctx' [-Wunused-parameter]
int tracepoint__syscalls__sys_enter_execve(struct trace_event_raw_sys_enter *ctx)
                                                                             ^
1 warning generated.
bpftool gen skeleton build/hello.bpf.o > build/hello.skel.h
clang -g -O2 -Wall -Wextra -I build -c hello.c -o build/hello.o
clang -g -O2 -Wall -Wextra build/hello.o -L../libbpf/build/libbpf -lbpf -lelf -lz -o build/hello
root@ubuntux86:# ./build/hello 

```