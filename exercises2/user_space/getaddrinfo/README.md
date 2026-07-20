# getaddrinfo

This is a simple program that calls
[getaddrinfo(3)](http://man7.org/linux/man-pages/man3/getaddrinfo.3.html)
and displays the results.

## Usage

To build do:

    make

And then

    ./getaddrinfo <hostname> <port number or service name>

E.g.

    $ ./getaddrinfo google.com http
    AF_INET SOCK_STREAM tcp 216.58.213.78 80 google.com
    AF_INET SOCK_DGRAM udp 216.58.213.78 80
    AF_INET6 SOCK_STREAM tcp 2a00:1450:4009:810::200e 80
    AF_INET6 SOCK_DGRAM udp 2a00:1450:4009:810::200e 80
	
	
# test2
```Shell
root@ubuntu:~# ./getaddrinfo   google.com http
AF_INET SOCK_STREAM tcp 142.251.42.238 80 google.com
root@ubuntu:~# ./getaddrinfo   8.8.8.8 53
AF_INET SOCK_STREAM tcp 8.8.8.8 53 8.8.8.8
AF_INET SOCK_DGRAM udp 8.8.8.8 53 
AF_INET <unknown socktype> ip 8.8.8.8 53 
```

```Shell
./getaddrinfo 127.0.0.1 22
AF_INET SOCK_STREAM tcp 127.0.0.1 22 127.0.0.1
AF_INET SOCK_DGRAM udp 127.0.0.1 22 
AF_INET <unknown socktype> ip 127.0.0.1 22 
```
```Shell
./getaddrinfo 10.11.11.251 22
AF_INET SOCK_STREAM tcp 10.11.11.251 22 10.11.11.251
AF_INET SOCK_DGRAM udp 10.11.11.251 22 
AF_INET <unknown socktype> ip 10.11.11.251 22 
```

