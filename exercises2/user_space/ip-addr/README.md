



```
>>> import ipaddress
>>> ipaddress.ip_network(u'10.220.192.192/29')
IPv4Network('10.220.192.192/29')
>>> 
```


```
>>> myipv6 = ipaddress.ip_address(u'ff05::1:3')
>>> print(myipv6)
ff05::1:3
>>> myipv6.exploded
'ff05:0000:0000:0000:0000:0000:0001:0003'
>>> 
>>> print(myipv6.exploded.split(":"))
['ff05', '0000', '0000', '0000', '0000', '0000', '0001', '0003']
>>> 
```

```
>>> secs=myipv6.exploded.split(":")
>>> print(secs[1])
0000
>>> print(secs[0])
ff05
>>> 

```