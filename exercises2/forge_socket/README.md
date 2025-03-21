forge_socket is a Linux kernel module that allows inspection and modification
of TCP sockets from user space.

For example, a user space application can specify any arbitrary TCP state
(i.e. source/destination IP and port, sequence/acknowledgement numbers), and
request the module modify a socket to these parameters. Then, the application
can call send() and recv() as normal on this socket.

Furthermore, you can request the state of a TCP socket, for example, to learn
the sequence and acknowledgement numbers of a connection owned by the process.


# INSTALLING
----------

To install forge_socket, from the module directory, run:

    make
    sudo insmod forge_socket.ko

To remove, run:
    
    sudo rmmod forge_socket 


# USING
-----

forge_socket provides userspace processes with an additional socket type,
SOCK_FORGE. You can instantiate with
    
    #include "forge_socket.h"

    ...

    int sock = socket(AF_INET, SOCK_FORGE, 0);

This socket will be identical to a TCP socket (SOCK_STREAM), but supports an
additional getsockopt and setsockopt option, TCP_STATE.

    int len;
    struct tcp_state state;
    getsockopt(sock, IPPROTO_TCP, TCP_STATE, &state, &len);

will fetch the current TCP state of the socket, and fill in members of the
tcp_state struct (defined in forge_socket.h)

    setsockopt(sock, IPPROTO_TCP, TCP_STATE, &state, sizeof(state));

will set the current state of a socket. Note that currently, this only
supports going from a TCP_CLOSE to a TCP_ESTABLISHED state socket, and it
assumes the userspace process has previously called bind(2) on the socket, to
set up some of the state. Additional options must also be set prior to this
(SO_REUSEADDR, and IP_TRANSPARENT) in some cases. Eventually this may become
part of the setsockopt call, for now, the caller is responsible for setting
these options. See set_sock_state() in test.c for example usage.

# test

+ client
```
insmod forge_socket.ko 
 ./client_win2 
```
+ server

```
./server_ip4_win 
```

