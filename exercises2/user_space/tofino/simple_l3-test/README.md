


```
bfrt.simple_l3.pipe.Ingress.ipv4_lpm> add_with_l3_switch(ip_address('10.10.15.134'), 32, port=160, new_mac_da=0x74a4b500eee5)

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> add_with_l3_switch(ip_address('10.10.15.135'), 32, port=168, new_mac_da=0x74a4b500f009)

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> dump
------------------------------------> dump()
----- ipv4_lpm Dump Start -----
Default Entry:
Entry data (action : Ingress.drop):

Entry 0:
Entry key:
    hdr.ipv4.dst_addr              : ('0x0A0A0F86', '0x00000020')
Entry data (action : Ingress.l3_switch):
    port                           : 0xA0
    new_mac_da                     : 0x74A4B500EEE5

Entry 1:
Entry key:
    hdr.ipv4.dst_addr              : ('0x0A0A0F87', '0x00000020')
Entry data (action : Ingress.l3_switch):
    port                           : 0xA8
    new_mac_da                     : 0x74A4B500F009

----- ipv4_lpm Dump End -----


bfrt.simple_l3.pipe.Ingress.ipv4_lpm> 
```