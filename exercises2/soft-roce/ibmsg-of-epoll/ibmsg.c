#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <rdma/rdma_verbs.h>

#include "ibmsg.h"
#include "utility.h"


int
ibmsg_init_event_loop(ibmsg_event_loop* event_loop)
{
    event_loop->event_channel = rdma_create_event_channel();

    int fd = event_loop->event_channel->fd;
    if(make_nonblocking(fd))
        return IBMSG_FCNTL_FAILED;

    event_loop->epollfd = epoll_create(IBMSG_MAX_EVENTS);
    if (event_loop->epollfd == -1) {
        return IBMSG_EPOLL_FAILED;
    }

    struct epoll_event ev;
    ev.events = EPOLLIN;
    event_loop->event_description.type = IBMSG_CMA;
    ev.data.ptr = &(event_loop->event_description);
    if (epoll_ctl(event_loop->epollfd, EPOLL_CTL_ADD, fd, &ev) == -1)
    {
        return IBMSG_EPOLL_ADD_FD_FAILED;
    }

    event_loop->connection_established = NULL;
    event_loop->connection_request = NULL;
    event_loop->message_received = NULL;

    return IBMSG_OK;
}


int
ibmsg_destroy_event_loop(ibmsg_event_loop* event_loop)
{
    rdma_destroy_event_channel(event_loop->event_channel);
    close(event_loop->epollfd);
    return IBMSG_OK;
}


int
ibmsg_connect(ibmsg_event_loop* event_loop, ibmsg_socket* connection, char* ip, unsigned short port)
{
    struct rdma_addrinfo* address_results;                                                                                                                         
    struct sockaddr_in dst_addr;                                                                                                                                   
    char service[16];
    snprintf(service, 16, "%d", port);

    connection->status = IBMSG_UNCONNECTED;
    connection->socket_type = IBMSG_SEND_SOCKET;

    memset(&dst_addr, 0, sizeof dst_addr);                                                                                                                         
    dst_addr.sin_family = AF_INET;                                                                                                                                 
    dst_addr.sin_port = htons(atoi(service));
    inet_pton(AF_INET, ip, &dst_addr.sin_addr);

    CHECK_CALL( rdma_getaddrinfo(ip, service, NULL, &address_results), IBMSG_GETADDRINFO_FAILED );
    CHECK_CALL( rdma_create_id (event_loop->event_channel, &connection->cmid, connection, RDMA_PS_TCP), IBMSG_CREATE_ID_FAILED );
    CHECK_CALL( rdma_resolve_addr (connection->cmid, NULL, (struct sockaddr*)&dst_addr, IBMSG_TIMEOUT_MS), IBMSG_ADDRESS_RESOLUTION_FAILED );
     
    return IBMSG_OK;
}


int ibmsg_disconnect(ibmsg_socket* connection)
{
    connection->status = IBMSG_DISCONNECTING;
    CHECK_CALL( rdma_disconnect(connection->cmid), IBMSG_DISCONNECT_FAILED );
    return IBMSG_OK;
}


int
ibmsg_alloc_msg(ibmsg_buffer* msg, ibmsg_socket* connection, size_t size)
{
    void* buffer = malloc(size);
    if(!buffer)
        return IBMSG_ALLOC_FAILED;
    msg->status = IBMSG_WAITING;
    msg->data = buffer;
    msg->size = size;
    msg->mr = rdma_reg_msgs(connection->cmid, buffer, size);
    if(!msg->mr)
    {
        LOG("Could not allocate message: %s", strerror(errno));
        free(buffer);
        return IBMSG_MEMORY_REGISTRATION_FAILED;
    }
    return IBMSG_OK;
}


int
ibmsg_free_msg(ibmsg_buffer* msg)
{
    int result = IBMSG_OK;
    if(rdma_dereg_mr(msg->mr))
        result = IBMSG_MEMORY_REGISTRATION_FAILED;
    free(msg->data);
    return result;
}


int
ibmsg_post_send(ibmsg_socket* connection, ibmsg_buffer* msg)
{
    CHECK_CALL( rdma_post_send(connection->cmid, msg /* wrid */, msg->data, msg->size, msg->mr, 0), IBMSG_POST_SEND_FAILED );
    return IBMSG_OK;
}


int
ibmsg_listen(ibmsg_event_loop* event_loop, ibmsg_socket* socket, char* ip, short port, int max_connections)
{
    struct sockaddr_in src_addr;

    src_addr.sin_family = AF_INET;
    src_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &src_addr.sin_addr);

    socket->socket_type = IBMSG_LISTEN_SOCKET;

    CHECK_CALL( rdma_create_id (event_loop->event_channel, &socket->cmid, NULL, RDMA_PS_TCP), IBMSG_CREATE_ID_FAILED );
    CHECK_CALL( rdma_bind_addr (socket->cmid, (struct sockaddr*)&src_addr), IBMSG_BIND_FAILED );
    CHECK_CALL( rdma_listen (socket->cmid, max_connections), IBMSG_LISTEN_FAILED );

    return IBMSG_OK;
}


int
ibmsg_accept(ibmsg_connection_request* request, ibmsg_socket* connection)
{
    struct ibv_qp_init_attr qp_init_attr;
    struct rdma_conn_param conn_param;

    init_qp_param(&qp_init_attr);
    CHECK_CALL( rdma_create_qp (request->cmid, NULL, &qp_init_attr), IBMSG_CREATE_QP_FAILED );

    init_conn_param(&conn_param);
    CHECK_CALL( rdma_accept (request->cmid, &conn_param), IBMSG_ACCEPT_FAILED );

    request->status = IBMSG_ACCEPTED;
    connection->cmid = request->cmid;
    connection->cmid->context = connection;
    connection->socket_type = IBMSG_RECV_SOCKET;
    return IBMSG_OK;
}

