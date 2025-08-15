


# kernel   
```
[root@localhost uloop3]# cat boot/config-5.3.0-18-generic | grep IPV6_SEG6
CONFIG_IPV6_SEG6_LWTUNNEL=y
CONFIG_IPV6_SEG6_HMAC=y
CONFIG_IPV6_SEG6_BPF=y
[root@localhost uloop3]# cat boot/config-5.3.0-18-generic | grep MATCH_SRH
CONFIG_IP6_NF_MATCH_SRH=m
```