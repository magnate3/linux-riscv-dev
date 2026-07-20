# netfilter_inspect

Utility that tracks netfilter DROP verdicts through the different namespaces, and displays the ethernet adapter properties. The objective is to be able to identify what adapter drops a packet, useful in a multi-adapter/multi-namespace environment.

## Output

The output can be grabbed via dmesg and will look something like this

```  
[Sun Jan  7 17:27:34 2024] ipt_do_table(filter) - devin=(null)/0, devout=eth0/2, saddr=0xa010002, daddr=0xa010001, proto=6, spt=0xb986, dpt=0x1f90, verdict=0
```

- devin: ingress device
- devout: egress device
- saddr: source IP address in little-endian (hex)
- daddr: destination IP address in little-endian (hex)
- proto: 6 = TCP and 17 = UDP (only supported protocols for now)
- spt: source TCP/UDP port in little-endian (hex)
- dpt: destination TCP/UDP port in little-endian (hex)
- retval: netfilter verdict NF_* which values/code mappings can be found in uapi/linux/netfilter.h
