

# 依赖/tools/lib/bpf/改为libpf

```
xdpsock_user.c:461:2: warning: implicit declaration of function ‘bpf_set_link_xdp_fd’; did you mean ‘bpf_link__fd’? [-Wimplicit-function-declaration]
  bpf_set_link_xdp_fd(opt_ifindex, -1, opt_xdp_flags);
  ^~~~~~~~~~~~~~~~~~~
  bpf_link__fd
```

```
xdpsock_user.c:593:6: warning: implicit declaration of function ‘bpf_prog_load_xattr’; did you mean ‘bpf_prog_load’? [-Wimplicit-function-declaration]
  if (bpf_prog_load_xattr(&prog_load_attr, &obj, &prog_fd))
```

+  1 bpf_prog_load_xattr 换成    bpf_object__open_file ……     

+ 2   bpf_xdp_attach换成 bpf_xdp_attach   

# xdpsock_user.c:xdp_umem_configure:332: Assertion failed: setsockopt(sfd, SOL_XDP, XDP_UMEM_REG, &mr, sizeof(mr)) == 0: errno: 22/"Invalid argument"

```
```