# krdma

RDMA easy to use in kernel.

`struct krdma_cb`: control block that supports both RDMA send/recv and read/write

## RDMA SEND/RECV APIs
```c
int krdma_send(struct krdma_cb *cb, const char *buffer, size_t length);

int krdma_receive(struct krdma_cb *cb, char *buffer);

/* Called with remote host & port */
int krdma_connect(const char *host, const char *port, struct krdma_cb **conn_cb);

/* Called with local host & port */
int krdma_listen(const char *host, const char *port, struct krdma_cb **listen_cb);

int krdma_accept(struct krdma_cb *listen_cb, struct krdma_cb **accept_cb);
```

KRDMA send receive example:

**node0**
```c
krdma_connect()
krdma_send()
krdma_receive()
krdma_send()
krdma_receive()
...
```

**node1**
```c
krdma_listen()
while (1) {
    krdma_accept()
    krdma_receive()
    krdma_send()
    krdma_receive()
    krdma_send()
}
```

## RDMA READ/WRITE APIs
```c
/* Called with remote host & port */
int krdma_rw_init_client(const char *host, const char *port, struct krdma_cb **cbp);

/* Called with local host & port */
int krdma_rw_init_server(const char *host, const char *port, struct krdma_cb **cbp);

int krdma_read(struct krdma_cb *cb, char *buffer, size_t length);

int krdma_write(struct krdma_cb *cb, const char *buffer, size_t length);

/* RDMA release API */
int krdma_release_cb(struct krdma_cb *cb);
```

KRDMA read write example:

**node0**
```c
krdma_rw_init_client()
krdma_write()
krdma_read()
krdma_write()
krdma_read()
...
```

**node1**
```c
krdma_rw_init_server()
while (1) {
    ...
}
```
