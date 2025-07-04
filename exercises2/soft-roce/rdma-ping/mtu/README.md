

#  ud    


```
ibv_ud_pingpong  -d mlx5_1 -g 3 -p 8777 -s 8192
Requested size larger than port MTU (4096)
```

```
 {
                struct ibv_port_attr port_info = {};
                int mtu;

                if (ibv_query_port(ctx->context, port, &port_info)) {
                        fprintf(stderr,
                                "Unable to query port info for port %d\n",
                                port);
                        goto clean_device;
                }
                mtu = 1 << (port_info.active_mtu + 7);
                if (size > mtu) {
                        fprintf(stderr,
                                "Requested size larger than port MTU (%d)\n",
                                mtu);
                        goto clean_device;
                }
        }
```