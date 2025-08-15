simpletun, a (too) simple tunnelling program.

-------

To compile the program, just do

$ gcc simpletun.c -o simpletun

If you have GNU make, you can also exploit implicit targets and do

$ make simpletun

-------

Usage:
simpletun -i <ifacename> [-s|-c <serverIP>] [-p <port>] [-u|-a] [-d]
simpletun -h

-i <ifacename>: Name of interface to use (mandatory)
-s|-c <serverIP>: run in server mode (-s), or specify server address (-c <serverIP>) (mandatory)
-p <port>: port to listen on (if run in server mode) or to connect to (in client mode), default 55555
-u|-a: use TUN (-u, default) or TAP (-a)
-d: outputs debug information while running
-h: prints this help text

-------

Refer to http://backreference.org/2010/03/27/tuntap-interface-tutorial/ for 
more information on tun/tap interfaces in Linux in general, and on this 
program in particular.
The program must be run at one end as a server, and as client at the other 
end. The tun/tap interface must already exist, be up and configured with an IP 
address, and owned by the user who runs simpletun. That user must also have
read/write permission on /dev/net/tun. (Alternatively, you can run the
program as root, and configure the transient interfaces manually before
starting to exchange packets. This is not recommended)

Use is straightforward. On one end just run

[server]$ ./simpletun -i tun13 -s

at the other end run

[client]$ ./simpletun -i tun0 -c 10.2.3.4

where 10.2.3.4 is the remote server's IP address, and tun13 and tun0 must be 
replaced with the names of the actual tun interfaces used on the computers.
By default it assumes a tun device is being used (use -u to be explicit), and
-a can be used to tell the program that the interface is tap. 
By default it uses TCP port 55555, but you can change that by using -p (the 
value you use must match on the client and the server, of course). Use -d to 
add some debug information. Press ctrl-c on either side to exit (the other end
will exit too).

The program is very limited, so expect to be disappointed.


# test

## Can't write to Linux tun/tap device (tun mode) /dev/net/tun: Input/output error

But one question here is if I don’t add "/sbin/ifconfig myvpn 10.0.0.1 netmask 255.255.255.0”, it seems the crontab wouldn’t trigger tinc-up, and then the ip addr of myvpn wouldn’t be configured, then it will prompt the error of "Can't write to Linux tun/tap device (tun mode)
 /dev/net/tun: Input/output error”
 
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/client.png)




```
# A主机
# 编译 simpletun
gcc simpletun.c -Wall -o vpn
# 作为 vpn server 启动，并开启 debug，默认监听 55555
sudo ./vpn -i tun0 -s -d
# 配置 tun 网卡地址
sudo ifconfig tun0 192.168.0.10 netmask 255.255.255.0

# B主机
# 编译 simpletun
gcc simpletun.c -Wall -o vpn
# 作为 vpn client 启动，连接 server，并开启 debug
sudo ./vpn -i tun0 -c 10.11.33.50 -d
# 配置 tun 网卡地址
sudo ifconfig tun0 192.168.0.11 netmask 255.255.255.0
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/ping.png)


## server

```
[root@bogon simpletun]# ./simpletun -i tun0 -s -d
Successfully connected to interface tun0
SERVER: Client connected from 10.10.16.251
TAP2NET 1: Read 76 bytes from the tap interface
TAP2NET 1: Written 76 bytes to the network
TAP2NET 2: Read 76 bytes from the tap interface
TAP2NET 2: Written 76 bytes to the network
TAP2NET 3: Read 76 bytes from the tap interface
TAP2NET 3: Written 76 bytes to the network
TAP2NET 4: Read 76 bytes from the tap interface
TAP2NET 4: Written 76 bytes to the network
NET2TAP 1: Read 48 bytes from the network
NET2TAP 1: Written 48 bytes to the tap interface
NET2TAP 2: Read 48 bytes from the network
NET2TAP 2: Written 48 bytes to the tap interface
NET2TAP 3: Read 48 bytes from the network
NET2TAP 3: Written 48 bytes to the tap interface
NET2TAP 4: Read 84 bytes from the network
NET2TAP 4: Written 84 bytes to the tap interface
TAP2NET 5: Read 84 bytes from the tap interface
TAP2NET 5: Written 84 bytes to the network
NET2TAP 5: Read 84 bytes from the network
NET2TAP 5: Written 84 bytes to the tap interface
TAP2NET 6: Read 84 bytes from the tap interface
TAP2NET 6: Written 84 bytes to the network
NET2TAP 6: Read 84 bytes from the network
NET2TAP 6: Written 84 bytes to the tap interface
TAP2NET 7: Read 84 bytes from the tap interface
TAP2NET 7: Written 84 bytes to the network
NET2TAP 7: Read 84 bytes from the network
NET2TAP 7: Written 84 bytes to the tap interface
TAP2NET 8: Read 84 bytes from the tap interface
TAP2NET 8: Written 84 bytes to the network
NET2TAP 8: Read 48 bytes from the network
NET2TAP 8: Written 48 bytes to the tap interface
```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/tcpdump.png)

## client

```
[root@centos7 simpletun]# ./simpletun  -i tun0 -c 10.10.16.81  -d
Successfully connected to interface tun0
CLIENT: Connected to server 10.10.16.81
TAP2NET 1: Read 48 bytes from the tap interface
TAP2NET 1: Written 48 bytes to the network
NET2TAP 1: Read 76 bytes from the network
NET2TAP 1: Written 76 bytes to the tap interface
TAP2NET 2: Read 48 bytes from the tap interface
TAP2NET 2: Written 48 bytes to the network
NET2TAP 2: Read 76 bytes from the network
NET2TAP 2: Written 76 bytes to the tap interface
NET2TAP 3: Read 76 bytes from the network
NET2TAP 3: Written 76 bytes to the tap interface
NET2TAP 4: Read 76 bytes from the network
NET2TAP 4: Written 76 bytes to the tap interface
TAP2NET 3: Read 48 bytes from the tap interface
TAP2NET 3: Written 48 bytes to the network
TAP2NET 4: Read 84 bytes from the tap interface
TAP2NET 4: Written 84 bytes to the network
NET2TAP 5: Read 84 bytes from the network
NET2TAP 5: Written 84 bytes to the tap interface
TAP2NET 5: Read 84 bytes from the tap interface
TAP2NET 5: Written 84 bytes to the network
NET2TAP 6: Read 84 bytes from the network
NET2TAP 6: Written 84 bytes to the tap interface
TAP2NET 6: Read 84 bytes from the tap interface
TAP2NET 6: Written 84 bytes to the network
NET2TAP 7: Read 84 bytes from the network
NET2TAP 7: Written 84 bytes to the tap interface
TAP2NET 7: Read 84 bytes from the tap interface
TAP2NET 7: Written 84 bytes to the network
NET2TAP 8: Read 84 bytes from the network
NET2TAP 8: Written 84 bytes to the tap interface
TAP2NET 8: Read 48 bytes from the tap interface
TAP2NET 8: Written 48 bytes to the network
TAP2NET 9: Read 48 bytes from the tap interface
TAP2NET 9: Written 48 bytes to the network

```

## client

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/client2.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/client3.png)


### ip

```
 struct iphdr *iph;
 struct sockaddr_in src, dst;
 iph = (struct iphdr *)buffer;
 src.sin_addr.s_addr = iph->saddr;
 dst.sin_addr.s_addr = iph->daddr;
 do_debug("################### NET2TAP src ip : %s, and dst ip : %s  \n", inet_ntoa(src.sin_addr),  inet_ntoa(dst.sin_addr));
```

因为数据流量没有经过链路层，所以数据包中是没有ethhdr的，也就是没有以太网帧，那么数据就是直接从iphdr开始，因此通过强制类型转换就可以直接从数据中提取需要的值

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/simpletun/ip.png)

```
        #include <stdio.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	int main()
	{
		char ip[] = "192.168.0.101";

		struct in_addr myaddr;
		/* inet_aton */
		int iRet = inet_aton(ip, &myaddr);
		printf("%x\n", myaddr.s_addr);

		/* inet_addr */
		printf("%x\n", inet_addr(ip));

		/* inet_pton */
		iRet = inet_pton(AF_INET, ip, &myaddr);
		printf("%x\n", myaddr.s_addr);

		myaddr.s_addr = 0xac100ac4;
		/* inet_ntoa */
		printf("%s\n", inet_ntoa(myaddr));

		/* inet_ntop */
		inet_ntop(AF_INET, &myaddr, ip, 16);
		puts(ip);
		return 0;
	}
```
# references

[Tun/Tap interface tutorial](https://backreference.org/2010/03/26/tuntap-interface-tutorial/)

[VPN 原理以及实现](https://paper.seebug.org/1648/)


[[教程] 在 Windows 上使用 tun2socks 进行全局代理](https://tachyondevel.medium.com/%E6%95%99%E7%A8%8B-%E5%9C%A8-windows-%E4%B8%8A%E4%BD%BF%E7%94%A8-tun2socks-%E8%BF%9B%E8%A1%8C%E5%85%A8%E5%B1%80%E4%BB%A3%E7%90%86-aa51869dd0d)
