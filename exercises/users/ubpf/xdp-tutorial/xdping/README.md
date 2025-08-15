

# linux/tools/testing/selftests/bpf/xdping.c 移植


```
root@ubuntux86:# make
    CC       xdping
In file included from xdping.c:20:
../lib/install/include/bpf/bpf.h:556:60: warning: ‘struct bpf_link_info’ declared inside parameter list will not be visible outside of this definition or declaration
 LIBBPF_API int bpf_link_get_info_by_fd(int link_fd, struct bpf_link_info *info, __u32 *info_len);
                                                            ^~~~~~~~~~~~~
In file included from xdping.c:21:
../lib/install/include/bpf/libbpf.h:70:54: warning: ‘enum bpf_link_type’ declared inside parameter list will not be visible outside of this definition or declaration
 LIBBPF_API const char *libbpf_bpf_link_type_str(enum bpf_link_type t);
                                                      ^~~~~~~~~~~~~
xdping.c: In function ‘main’:
xdping.c:173:6: warning: implicit declaration of function ‘bpf_prog_test_load’; did you mean ‘bpf_prog_load’? [-Wimplicit-function-declaration]
  if (bpf_prog_test_load(filename, BPF_PROG_TYPE_XDP, &obj, &prog_fd)) {
      ^~~~~~~~~~~~~~~~~~
      bpf_prog_load
```