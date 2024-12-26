
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

# ipv4

```
>>> addr=ipaddress.ip_address('10.10.10.10') 
>>> print('0x%08x'%int(addr)) 
0x0a0a0a0a
>>> 

```

# ipv4 to ipv6

```
>>> import ipaddress
>>> a = '10.20.30.40'
>>> print(ipaddress.IPv6Address('2002::' + a).compressed)
2002::a14:1e28
>>> 


```

# ipv4/ipv6 and str


```
print(str(ipaddress.ip_address(0xdffff980)))
223.255.249.128
```

# ipv4/ipv6 and int
[ipv4/ipv6 and int](https://stackoverflow.com/questions/9590965/convert-an-ip-string-to-a-number-and-vice-versa)
```
>>> import ipaddress
>>> int(ipaddress.ip_address('1.2.3.4'))
16909060
>>> str(ipaddress.ip_address(16909060))
'1.2.3.4'
```


```
>>> import ipaddress 
>>> print(ipaddress.ip_address(0x100000000)) 
::1:0:0
>>> 
>>> print("0x%08x"%0x100000000) 
0x100000000
>>> 

```

In Python 3, ints and longs have been merged into just int, which functions pretty much like long used to.    
```
>>> int(ipaddress.ip_address(u'1000:2000:3000:4000:5000:6000:7000:8000'))
21268296984521553528558659310639415296
>>> str(ipaddress.ip_address(21268296984521553528558659310639415296L))
  File "<stdin>", line 1
    str(ipaddress.ip_address(21268296984521553528558659310639415296L))
                                                                   ^
SyntaxError: invalid syntax
>>> str(ipaddress.ip_address(21268296984521553528558659310639415296))
'1000:2000:3000:4000:5000:6000:7000:8000'
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