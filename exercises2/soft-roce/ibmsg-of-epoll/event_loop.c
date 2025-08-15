#include <stdlib.h>
#include <sys/epoll.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_verbs.h>
#include "ibmsg.h"
#include "utility.h"

static void
free_connection(ibmsg_socket* connection)
{
    rdma_destroy_qp(connection->cmid);
    rdma_destroy_id(connection->cmid);
    connection->cmid = NULL;
}


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


static int
remove_connection_from_epoll(int epollfd, ibmsg_socket* connection)
{
    int fd;

    fd = connection->cmid->send_cq_channel->fd;
    if( epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, NULL) == -1)
    {
        return -1;
    }

    fd = connection->cmid->recv_cq_channel->fd;
    if( epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, NULL) == -1)
    {
        return -1;
    }

    return 0;
}


static void
post_receive(ibmsg_socket* connection)
{
    ibmsg_buffer* msg = &connection->recv_buffer;
    if(ibmsg_alloc_msg(msg, connection, IBMSG_MAX_MSGSIZE) != IBMSG_OK)
    {
        LOG("could not allocate memory for connection receive buffer");
        connection->status = IBMSG_ERROR;
    }
    LOG( "RECV: WRID 0x%llx", (long long unsigned)msg );
    rdma_post_recv(connection->cmid, msg, msg->data, msg->size, msg->mr);
    LOG( "Message receive posted" );
}


static void
process_rdma_event(ibmsg_event_loop* event_loop, struct rdma_cm_event* event)
{
    ibmsg_socket* connection = (ibmsg_socket*)event->id->context;
    struct ibv_qp_init_attr qp_init_attr;
    struct rdma_conn_param conn_param;
    ibmsg_connection_request request;

    LOG("EVENT: %s", rdma_event_str(event->event));

    switch(event->event)
    {
        case RDMA_CM_EVENT_ADDR_RESOLVED:
            connection->status = IBMSG_ADDRESS_RESOLVED;
            init_qp_param(&qp_init_attr);
            if(rdma_create_qp (connection->cmid, NULL, &qp_init_attr))
            {
                LOG("Create QP failed: %d '%s;", errno, strerror(errno));
                connection->status = IBMSG_ERROR;
            }
            if(rdma_resolve_route (connection->cmid, IBMSG_TIMEOUT_MS))
            {
                LOG("Resolve route failed: %d '%s;", errno, strerror(errno));
                connection->status = IBMSG_ERROR;
            }
            rdma_ack_cm_event (event);
            break;
        case RDMA_CM_EVENT_ADDR_ERROR:
            connection->status = IBMSG_ERROR;
            LOG("Could not resolve address: %d '%s;", errno, strerror(errno));
            rdma_ack_cm_event (event);
            break;
        case RDMA_CM_EVENT_ROUTE_RESOLVED:
            connection->status = IBMSG_ROUTE_RESOLVED;
            memset(&conn_param, 0, sizeof conn_param);
            init_conn_param(&conn_param);
            if(rdma_connect(connection->cmid, &conn_param))
            {
                LOG("RDMA connect failed: %d '%s;", errno, strerror(errno));
                connection->status = IBMSG_ERROR;
            }
            rdma_ack_cm_event (event);
            break;
        case RDMA_CM_EVENT_ROUTE_ERROR:
            connection->status = IBMSG_ERROR;
            LOG("Could not route: %d '%s;", errno, strerror(errno));
            rdma_ack_cm_event (event);
            break;
        case RDMA_CM_EVENT_CONNECT_REQUEST:
            if(event_loop->connection_request)
            {
                request.cmid = event->id;
                request.status = IBMSG_UNDECIDED;
                event_loop->connection_request(&request);
            }
            if(request.status != IBMSG_ACCEPTED)
            {
                LOG("connection request rejected");
                rdma_reject(event->id, NULL, 0);                    
            }
            else
            {
                LOG("connection request accepted");
            }
            rdma_ack_cm_event (event);
            break;
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
        case RDMA_CM_EVENT_REJECTED:
            connection->status = IBMSG_ERROR;
            rdma_ack_cm_event (event);
            free_connection(connection);
            break;
        case RDMA_CM_EVENT_DISCONNECTED:
            connection->status = IBMSG_UNCONNECTED;
            rdma_ack_cm_event (event);
            remove_connection_from_epoll(event_loop->epollfd, connection);
            free_connection(connection);
            LOG("Connection closed");
            break;
        default:
            break;
    }
}


static int
process_send_socket_send_completion(ibmsg_socket* connection)
{
    struct ibv_wc wc;
    int n_completions = rdma_get_send_comp(connection->cmid, &wc);
    if(n_completions == -1)
    {
        LOG("send completion error: %d ('%s')", errno, strerror(errno));
        return IBMSG_FETCH_EVENT_FAILED;
    }

    ibmsg_buffer* msg = (ibmsg_buffer*) wc.wr_id;
    msg->status = IBMSG_SENT;
    LOG( "%d send completion(s) found", n_completions);
    LOG( "SEND complete: WRID 0x%llx", (long long unsigned)wc.wr_id );
    return 0;
}


static int
process_recv_socket_recv_completion(ibmsg_event_loop* event_loop, ibmsg_socket* connection)
{
    struct ibv_wc wc;
    int n_completions = rdma_get_recv_comp(connection->cmid, &wc);
    if(n_completions == -1)
    {
        LOG("recv completion error: %d ('%s')", errno, strerror(errno));
        return IBMSG_FETCH_EVENT_FAILED;
    }

    ibmsg_buffer* msg = (ibmsg_buffer*) (wc.wr_id);
    msg->status = IBMSG_RECEIVED;
    LOG( "%d recv completion(s) found", n_completions);
    LOG( "RECV complete: WRID 0x%llx", (long long unsigned)msg );
    if(event_loop->message_received)
    {
        event_loop->message_received(connection, msg);
    }
    else
    {
        if(ibmsg_free_msg(msg))
            return IBMSG_FREE_BUFFER_FAILED;
    }
    if(connection->status == IBMSG_CONNECTED)
        post_receive(connection);
    return 0;
}


int
ibmsg_dispatch_event_loop(ibmsg_event_loop* event_loop)
{
    int nfds;
    struct epoll_event events[IBMSG_MAX_EVENTS];
    do
    {
        nfds = epoll_wait(event_loop->epollfd, events, IBMSG_MAX_EVENTS, 10);
        if (nfds == -1) {
            return IBMSG_EPOLL_WAIT_FAILED;
        }
    } while(nfds == 0);

    LOG("Found %d event(s)", nfds);
    struct rdma_cm_event *cm_event;
    struct _ibmsg_event_description* data;
    ibmsg_socket* connection;
    int result;
    for(int i=0; i<nfds; i++)
    {
        data = events[i].data.ptr;
        switch(data->type)
        {
        case IBMSG_CMA:
            CHECK_CALL( rdma_get_cm_event (event_loop->event_channel, &cm_event), IBMSG_FETCH_EVENT_FAILED );
            process_rdma_event(event_loop, cm_event);
            break;
        case IBMSG_SEND_COMPLETION:
            connection = data->ptr;
            if(connection->socket_type == IBMSG_SEND_SOCKET)
            {
                if((result = process_send_socket_send_completion((ibmsg_socket*)data->ptr)))
                    return result; 
            }
            break;
        case IBMSG_RECV_COMPLETION:
            connection = data->ptr;
            if(connection->socket_type == IBMSG_RECV_SOCKET)
            {
                if((result = process_recv_socket_recv_completion(event_loop, (ibmsg_socket*)data->ptr)))
                    return result;
            }
            break;
        }
    }

    return IBMSG_OK;
}
