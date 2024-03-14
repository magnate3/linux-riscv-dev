
# test1

## server


![image](srv82.png)

## client

 

![image](srv156.png)

![image](srv251.png)


# test2  sendmsg to local server

## server 

![image](local.png)

## client

```
[root@centos7 c]# ./cli 
bind result 0 port 5005
sent 8 bytes
[root@centos7 c]#
```


# test3  sendmsg to remote server

remote server= 10.10.16.82:5000


![image](remote2.png)



```
[root@centos7 c]# gcc udp_pkt_cli.c -o cli
[root@centos7 c]# ./cli
bind result 0 port 5005
sent -1 bytes
error Invalid argument
[root@centos7 c]# 
```

## no pktinfo


### client

![image](remote_en.png)

![image](remote_en2.png)


### server

![image](remote_en3.png)


# references

[UDP Socket 编程](https://www.jianshu.com/p/22b32e5a267d)




 


