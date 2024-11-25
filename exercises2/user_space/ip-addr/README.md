
```
>>> print("0x%x"%a)
0x86
>>> 
>>> print("0x%x,%d"%(a,a*2))
0x19180,205568
>>> 
```


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


```
>>> print(myipv6.exploded)
2008:0000:0000:0000:0000:0000:0000:0004
>>> print(myipv6.exploded.replace(":",""))
20080000000000000000000000000004
>>> 
```


#   Convert IP address to integer and vice versa

```
# importing the module 
import ipaddress 
  
# converting IPv4 address to int 
addr1 = ipaddress.ip_address('191.255.254.40') 
addr2 = ipaddress.ip_address('0.0.0.123') 
print(int(addr1)) 
print(int(addr2)) 
  
# converting IPv6 address to int 
addr3 = ipaddress.ip_address('2001:db7:dc75:365:220a:7c84:d796:6401') 
print(int(addr3))
```

```
[root@centos7 demos]# python3 test.py 
3221225000
123
42540766400282592856903984001653826561
[root@centos7 demos]# 
```

## test2
```
[root@centos7 demos]# cat test2.py 
# importing the module 
import ipaddress 
# converting int to IPv4 address 
print(ipaddress.ip_address(3221225000)) 
print(ipaddress.ip_address(123)) 
# converting int to IPv6 address 
print(ipaddress.ip_address(42540766400282592856903984001653826561))
[root@centos7 demos]# 
```

```
[root@centos7 demos]# python3 test2.py 
191.255.254.40
0.0.0.123
2001:db7:dc75:365:220a:7c84:d796:6401
[root@centos7 demos]# 
```