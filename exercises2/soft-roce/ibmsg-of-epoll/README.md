# ibmsg
Asynchronous Infiniband Message Library

Ibmsg is an open source library providing a reliable messaging service in Infiniband networks.
It is a thin layer on top of librdma and libibverbs for connection handling and communication.

## Documentation

### Examples

Please see [ibmsg-send.c](ibmsg-send.c) and [ibmsg-recv.c](ibmsg-recv.c) for an example of how to use ibmsg.

#### server

```
[root@centos7 ibmsg]# ./ibmsg-recv 10.11.11.251
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_CONNECT_REQUEST
LOG: connection request accepted
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_ESTABLISHED
LOG: RECV: WRID 0x38917a60
LOG: Message receive posted
connection established
LOG: Found 1 event(s)
LOG: 1 recv completion(s) found
LOG: RECV complete: WRID 0x38917a60
message received
LOG: RECV: WRID 0x38917a60
LOG: Message receive posted
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_DISCONNECTED
LOG: Connection closed
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_CONNECT_REQUEST
LOG: connection request accepted
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_ESTABLISHED
LOG: RECV: WRID 0x38910340
LOG: Message receive posted
connection established
LOG: Found 1 event(s)
LOG: 1 recv completion(s) found
LOG: RECV complete: WRID 0x38910340
message received
LOG: RECV: WRID 0x38910340
LOG: Message receive posted
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_DISCONNECTED
LOG: Connection closed

```

#### client
```
root@ubuntu:~/the-geek-in-the-corner/ibmsg# ./ibmsg-send 10.11.11.251
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_ADDR_RESOLVED
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_ROUTE_RESOLVED
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_ESTABLISHED
LOG: RECV: WRID 0xfffff6d3d168
LOG: Message receive posted
LOG: Found 1 event(s)
LOG: 1 send completion(s) found
LOG: SEND complete: WRID 0xfffff6d3d100
LOG: Found 1 event(s)
LOG: Found 1 event(s)
LOG: EVENT: RDMA_CM_EVENT_DISCONNECTED
LOG: Connection closed
```

### Event Loop

Events in ibmsg are processed by an event loop. 
An event loop data structure is used to keep track of events:

    int ibmsg_init_event_loop(ibmsg_event_loop* event_loop)
    int ibmsg_destroy_event_loop(ibmsg_event_loop* event_loop)

The user is responsible to execute the event loop.
The event loop can either be run in its own thread in an endless loop, or mixed with 
other ibmsg operations. This functions processes standing events of the event loop
and then returns to the user when all events have been processed:

    int ibmsg_dispatch_event_loop(ibmsg_event_loop* event_loop)

The user may define several callbacks in an `ibmsg_event_loop` object that are
executed when certain events occur. Currently these callbacks function are:

    void (*connection_request)(ibmsg_connection_request*)
    void (*connection_established)(ibmsg_socket*)
    void (*message_received)(ibmsg_socket*, ibmsg_buffer*)



### Connection Management

Connections are tracked in a `ibmsg_socket` data structure.
Connections are used on the server as well as one the client side.
A server can listen on a specific IP address and port:

    int ibmsg_listen(ibmsg_event_loop* event_loop, ibmsg_socket* socket, char* ip, short port, int max_connections)

Once a client requests to open a connection with the server, the `connection_request` 
callback is executed on the server side. The user has to supply a callback function for
`connection_request` that either accepts or rejects the request. 
The new `ibmsg_socket` object is passed as parameter to the callback function. 
By default all requests are rejected if no callback function is supplied.
If a user wants to accept a connection,
the following function must be called from the callback:

    int ibmsg_accept(ibmsg_connection_request* request, ibmsg_socket* connection)

The function `ibmsg_connect` has to be evoked on the client side to connect to a remote host:

    int ibmsg_connect(ibmsg_event_loop* event_loop, ibmsg_socket* connection, char* ip, unsigned short port)
    
Either client or server can close a connection at all times using `ibmsg_disconnect`:
    
    int ibmsg_disconnect(ibmsg_socket* connection);


### Messages

Messages are enclosed in `ibmsg_buffer` objects and have to allocated and freed with the following
functions:

    int ibmsg_alloc_msg(ibmsg_buffer* msg, ibmsg_socket* connection, size_t size)
    int ibmsg_free_msg(ibmsg_buffer* msg)
    
To send a message to the remote side of a connection, call `ibmsg_post_send`. Remember that ibmsg is
asynchronous: the call will return immediately even if the message is still being sent. Do not
attempt to free the message before the status of the message changed to IBMSG_SENT.

    int ibmsg_post_send(ibmsg_socket* connection, ibmsg_buffer* msg)

### Error Handling

All functions of the ibmsg API return an error code as integer. A return code of 0 (or IBMSG_OK) indicates
success. Any other return code indicates an error. See ibmsg.h for the different error codes.

### epoll

```
struct rdma_cm_id {
        struct ibv_context      *verbs;
        struct rdma_event_channel *channel;
        void                    *context;
        struct ibv_qp           *qp;
        struct rdma_route        route;
        enum rdma_port_space     ps;
        uint8_t                  port_num;
        struct rdma_cm_event    *event;
        struct ibv_comp_channel *send_cq_channel;
        struct ibv_cq           *send_cq;
        struct ibv_comp_channel *recv_cq_channel;
        struct ibv_cq           *recv_cq;
        struct ibv_srq          *srq;
        struct ibv_pd           *pd;
        enum ibv_qp_type        qp_type;
};

```

```
 case RDMA_CM_EVENT_ESTABLISHED:
            if(add_connection_to_epoll(event_loop->epollfd, connection))
            {
                connection->status = IBMSG_ERROR;
                free_connection(connection);
            }
            post_receive(connection);
            rdma_ack_cm_event (event);
            connection->status = IBMSG_CONNECTED;
            if(event_loop->connection_established)
                event_loop->connection_established(connection);
            break;
```


```
static int
add_connection_to_epoll(int epollfd, ibmsg_socket* connection)
{
    struct epoll_event ev;
    int fd;

    if(ibv_req_notify_cq(connection->cmid->send_cq, 0))
        return -1;
    fd = connection->cmid->send_cq_channel->fd;
    make_nonblocking(fd);
    connection->send_event_description.type = IBMSG_SEND_COMPLETION;
    connection->send_event_description.ptr = connection;
    ev.events = EPOLLIN | EPOLLET;
    ev.data.ptr = &(connection->send_event_description);
    if (epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &ev) == -1)
    {
        return -1;
    }

    if(ibv_req_notify_cq(connection->cmid->recv_cq, 0))
        return -1;
    fd = connection->cmid->recv_cq_channel->fd;
    make_nonblocking(fd);
    connection->recv_event_description.type = IBMSG_RECV_COMPLETION;
    connection->recv_event_description.ptr = connection;
    ev.events = EPOLLIN | EPOLLET;
    ev.data.ptr = &(connection->recv_event_description);
    if (epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &ev) == -1)
    {
        return -1;
    }

    return 0;
}
```

#### send 事件

```
        case IBMSG_SEND_COMPLETION:
            connection = data->ptr;
            if(connection->socket_type == IBMSG_SEND_SOCKET)
            {
                if((result = process_send_socket_send_completion((ibmsg_socket*)data->ptr)))
                    return result;
            }
            break;
```

```
(gdb) bt
#0  process_send_socket_send_completion (connection=0xfffffffff918) at event_loop.c:187
#1  0x0000aaaaaaaad824 in ibmsg_dispatch_event_loop (event_loop=0xfffffffff8e0) at event_loop.c:265
#2  0x0000aaaaaaaab8a0 in main (argc=2, argv=0xfffffffffaa8) at ibmsg-send.c:66
(gdb) 
```

#### recv 事件

```

```

### rdma 事件

```
(gdb) bt
#0  process_rdma_event (event_loop=0xfffffffff8e0, event=0xaaaaaaacabf0) at event_loop.c:91
#1  0x0000aaaaaaaad7f8 in ibmsg_dispatch_event_loop (event_loop=0xfffffffff8e0) at event_loop.c:259
#2  0x0000aaaaaaaab7d0 in main (argc=2, argv=0xfffffffffaa8) at ibmsg-send.c:42
```
