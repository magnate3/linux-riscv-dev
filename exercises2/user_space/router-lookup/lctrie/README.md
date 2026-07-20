# LC-Trie, Level Compressed Tries for IPv4 Subnet Matching
[Linux路由子系统——LC-Trie](https://www.acwing.com/blog/content/23314/)   
This project provides a level compressed trie C library
for the purposes of matching IP addresses to arbitrarily
define IPv4 CIDR subnets.  Various address types are
supported, and the library has the RFC 1918 private IPv4
address space as well as the remaining RFC 5735 special
IPv4 address ranges hard coded into an auxiliary IPv4
information module.

For our purposes, it should be sufficient to dump the trie
and start from scratch again if we want to rebuild the trie.
This will correspond to a front/back buffer should this need
to be wired into a performance critical asynchronous system.

--

## Fast IP Routing with LC-Tries    
Based on the paper "Fast address lookup for Internet routers" by Stefan Nilsson and Gunnar Karlsson. This is the reference code for the Dr.Dobb's article from Aug 1998.   

You can also find parts of the trie used in the linux kernel (fib_trie.c)   
## Instructions


### prefix item

```
#if LCT_VERIFY_PREFIXES
  // Add a couple of custom prefixes.  Just use the void *data as a char *desc

  // 192.168.1.0/24 home subnet (common for SOHO wireless routers)
  p[num].info.type = IP_SUBNET_USER;
  p[num].info.usr.data = "Class A/24 home network";
  inet_pton(AF_INET, "192.168.1.0", &(p[num].addr));
  p[num].addr = ntohl(p[num].addr);
  p[num].len = 24;
  ++num;

  // 192.168.1.0/28 home sub-subnet.  used for testing address ranges
  p[num].info.type = IP_SUBNET_USER;
  p[num].info.usr.data = "Class A/24 guest network";
  inet_pton(AF_INET, "192.168.2.0", &(p[num].addr));
  p[num].addr = ntohl(p[num].addr);
  p[num].len = 24;
  ++num;

  // 192.168.1.0/28 home sub-subnet.  used for testing address ranges
  p[num].info.type = IP_SUBNET_USER;
  p[num].info.usr.data = "Class A/24 NAS network";
  inet_pton(AF_INET, "192.168.22.0", &(p[num].addr));
  p[num].addr = ntohl(p[num].addr);
  p[num].len = 24;
  ++num;
#endif
```

```
 qsort(p, num, sizeof(lct_subnet_t), subnet_cmp);
 lct_build(&t, p, num);
 subnet = lct_find(&t, ntohl(prefix));
```

### How to Build

Build requirements:
make, GCC, glibc, pcre

To build:
On any relatively modern unix system, simply typing make should
produce the lctrie_test executable binary.

### How to Run

```
[root@centos7 lctrie]# ./lctrie_test bgp/data-raw-table
Reading prefixes from bgp/data-raw-table...

Subnet 192.88.99.0/24 type 7 duplicates another of type 1
1 duplicates removed
```

This will use the raw APNIC BGP prefix table, run some basic
tests against the library, and then conduct a 5 second performance
test against the library with randomized lookup addresses.

Performance metrics and runtime stastics will be produced at the
end of each runtime step.

--

## Copyright and License

This project is copyright 2016 Charles Stewart <chuckination@gmail.com>.

Software is licensed under [2-Clause BSD Licnese](https://github.com/chuckination/lctrie/blob/master/LICENSE).

--

### Bibliography

* BGP Routing Table Analysis - Washington, Asia Pacific Network Information Centre, 2016

--http://thyme.apnic.net/us/

* Stefan Nilsson and Gunnar Karlsson, IP-Address Lookup Using LC-Tries, KTH Royal Institute of Technology, 1998

--https://www.nada.kth.se/~snilsson/publications/IP-address-lookup-using-LC-tries/text.pdf

* Stefan Nilsson and Gunnar Karlsson, Fast IP Routing with LC-Tries, Dr. Dobb's, 1998

--http://www.drdobbs.com/cpp/fast-ip-routing-with-lc-tries/184410638

* Weidong Wu, Packet Forwarding Technologies, CRC Press, 2007

* RFC 1519, Classless Inter-Domain Routing (CIDR), IETF, 1993

--https://tools.ietf.org/html/rfc1519

* RFC 4632, Classless Inter-Domain Routing (CIDR), IETF, 2006

--https://tools.ietf.org/html/rfc4632

* RFC 5735, Special Use IPv4 Addresses, IETF, 2010

--https://tools.ietf.org/html/rfc5735

* RFC 1918, Address Allocation for Private Internets, IETF, 1996

--https://tools.ietf.org/html/rfc1918
