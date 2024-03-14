

```Shell
gcc -Wstrict-prototypes -Wall -Wextra -Wno-sign-compare -Wpointer-arith -Wformat -Wformat-security -Wswitch-enum -Wunused-parameter -Wbad-function-cast -Wcast-align -Wstrict-prototypes -Wold-style-definition -Wmissing-prototypes -Wmissing-field-initializers -fno-strict-aliasing -Wswitch-bool -Wlogical-not-parentheses -Wsizeof-array-argument -Wbool-compare -Wshift-negative-value -Wduplicated-cond -Wshadow -Wmultistatement-macros -Wcast-align=strict -I/work/ovs_p4/xdp_proj/bpf_install/usr/local/include -o utilities/ovs-appctl utilities/ovs-appctl.o  lib/.libs/libopenvswitch.a -lelf -latomic -lnuma -lpthread -lrt -lm -L/work/ovs_p4/xdp_proj/bpf_install/usr/local/lib64 -lbpf -Wl,-rpath=/work/ovs_p4/xdp_proj/bpf_install/usr/local/lib64 -lelf -lunwind 
```
`-L/work/ovs_p4/xdp_proj/bpf_install/usr/local/lib64 -lbpf -Wl,-rpath=/work/ovs_p4/xdp_proj/bpf_install/usr/local/lib64 -lelf`

-Wl,-rpath=<your_lib_dir> 为程序添加一个运行时库文件搜索路径  
-Wl：这个是gcc的参数，表示编译器将后面的参数传递给链接器 ld。   
-rpath：添加一个文件夹作为运行时库的搜索路径。在运行链接时，优先搜索-rpath路径，再去搜索LD_RUN_PATH路径。   
例如：   
# 指定链接的位置 $(prefix)/lib。程序运行时会先去$(prefix)/lib下搜索所需库文件。
gcc -o foo foo.c -L$(prefix)/lib -lfoo -Wl,-rpath=$(prefix)/lib   


# static

echo 'LDLIBS += -L /work/xdp-tutorial/lib/install/lib/usr/lib64 -l:libbpf.a' >>$CONFIG
