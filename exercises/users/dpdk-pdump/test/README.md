

```
[root@centos7 test]# gcc test_pcap.c  -o test_pcap
[root@centos7 test]# ./test_pcap 
```

```
[root@centos7 test]# tcpdump  -nr test.pcap 
reading from file test.pcap, link-type EN10MB (Ethernet)
-4:00:00.000000 IP 192.168.58.128.59642 > 91.189.91.38.http: Flags [S], seq 1540076386, win 64240, options [mss 1460,sackOK,TS val 2083233854 ecr 0,nop,wscale 7], length 0
[root@centos7 test]# 
```