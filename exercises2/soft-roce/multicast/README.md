

# mckey
qp_type
```
 init_qp_attr.qp_type = IBV_QPT_UD
```


```
mckey -b  10.11.11.251 -m 239.1.2.1
mckey: starting server
mckey: joining
mckey: joined dgid: ::ffff:239.1.2.1 mlid 0x0 sl 0
receiving data transfers

```


```
 mckey -b 10.11.11.82 -m 239.1.2.2 -s  -C 1024 -S 64
mckey: starting client
mckey: joining
mckey: joined dgid: ::ffff:239.1.2.2 mlid 0x0 sl 0
initiating data transfers
data transfers complete
test complete
return status 0
```

# references

[Linux Multicast 配置](https://www.jianshu.com/p/9c5540a26e4f)