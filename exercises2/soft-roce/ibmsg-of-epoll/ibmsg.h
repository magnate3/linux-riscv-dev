#ifndef IBMSG_RDMA_H
#define IBMSG_RDMA_H

#include <rdma/rdma_cma.h>

#define IBMSG_MAX_MSGSIZE       (4096)


enum
{
    IBMSG_OK,
    IBMSG_FCNTL_FAILED,
    IBMSG_GETADDRINFO_FAILED,
    IBMSG_CREATE_ID_FAILED,
    IBMSG_BIND_FAILED,
    IBMSG_LISTEN_FAILED,
    IBMSG_ADDRESS_RESOLUTION_FAILED,
    IBMSG_FETCH_EVENT_FAILED,
    IBMSG_ACK_EVENT_FAILED,
    IBMSG_ALLOC_FAILED,
    IBMSG_MEMORY_REGISTRATION_FAILED,
    IBMSG_POST_SEND_FAILED,
    IBMSG_DISCONNECT_FAILED,
    IBMSG_FREE_BUFFER_FAILED,
    IBMSG_CREATE_QP_FAILED,
    IBMSG_ACCEPT_FAILED,
    IBMSG_EPOLL_FAILED,
    IBMSG_EPOLL_ADD_FD_FAILED,
    IBMSG_EPOLL_WAIT_FAILED
};


typedef struct
{
    struct rdma_cm_id* cmid;
    enum {
        IBMSG_UNDECIDED, IBMSG_ACCEPTED
    } status;

} ibmsg_connection_request;

typedef struct
{
    struct ibv_mr* mr;
    void* data;
    size_t size;
    enum {
        IBMSG_WAITING, IBMSG_SENT, IBMSG_RECEIVED
    } status;
} ibmsg_buffer;

struct _ibmsg_event_description 
{
    /* used internally */
    enum { 
        IBMSG_CMA, IBMSG_SEND_COMPLETION, IBMSG_RECV_COMPLETION
    } type;
    void* ptr;
};

typedef struct
{
    struct rdma_cm_id* cmid;
    enum {
        IBMSG_UNCONNECTED,
        IBMSG_ADDRESS_RESOLVED,
        IBMSG_ROUTE_RESOLVED,
        IBMSG_CONNECTED,
        IBMSG_DISCONNECTING,
        IBMSG_ERROR
    } status;
    enum {
        IBMSG_LISTEN_SOCKET,
        IBMSG_RECV_SOCKET,
        IBMSG_SEND_SOCKET
    } socket_type;
    ibmsg_buffer recv_buffer;

    struct _ibmsg_event_description send_event_description;
    struct _ibmsg_event_description recv_event_description;
} ibmsg_socket;

typedef struct
{
    struct rdma_event_channel* event_channel;
    int epollfd;

    /* callbacks */
    void (*connection_request)(ibmsg_connection_request*);
    void (*connection_established)(ibmsg_socket*);
    void (*message_received)(ibmsg_socket*, ibmsg_buffer*);

    struct _ibmsg_event_description event_description;
} ibmsg_event_loop;


int ibmsg_init_event_loop(ibmsg_event_loop* event_loop);
int ibmsg_destroy_event_loop(ibmsg_event_loop* event_loop);
int ibmsg_connect(ibmsg_event_loop* event_loop, ibmsg_socket* connection, char* ip, unsigned short port);
int ibmsg_disconnect(ibmsg_socket* socket);
int ibmsg_listen(ibmsg_event_loop* event_loop, ibmsg_socket* socket, char* ip, short port, int max_connections);
int ibmsg_accept(ibmsg_connection_request* request, ibmsg_socket* connection);
int ibmsg_alloc_msg(ibmsg_buffer* msg, ibmsg_socket* connection, size_t size);
int ibmsg_free_msg(ibmsg_buffer* msg);
int ibmsg_post_send(ibmsg_socket* connection, ibmsg_buffer* msg);
int ibmsg_dispatch_event_loop(ibmsg_event_loop* event_loop);

#endif
