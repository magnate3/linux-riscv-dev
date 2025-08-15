#include <iostream>
#include <thread>

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

const int BUFFER_SIZE = 2048;
const int TIMEOUT_IN_MS = 500; /* ms */

struct context
{
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_comp_channel *comp_channel;

    pthread_t cq_poller_thread;
};

struct connection
{
    struct rdma_cm_id *id;
    struct ibv_qp *qp;

    struct ibv_mr *recv_mr;
    struct ibv_mr *send_mr;

    char *recv_region;
    char *send_region;

    int num_completions;
};

static pthread_t msgThread;

static void die(const char *reason);

static void build_context(struct ibv_context *verbs);
static void build_qp_attr(struct ibv_qp_init_attr *qp_attr);
static void * poll_cq(void *);
static void post_receives(struct connection *conn);
static void register_memory(struct connection *conn);

static int on_addr_resolved(struct rdma_cm_id *id);
static void on_completion(struct ibv_wc *wc);
static int on_connection(void *context);
static int on_disconnect(struct rdma_cm_id *id);
static int on_event(struct rdma_cm_event *event);
static int on_route_resolved(struct rdma_cm_id *id);

static struct context *s_ctx = NULL;

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;

bool ok_to_send_next_message = 1;
bool message_available()
{
    return 0 != ok_to_send_next_message;
}

int main(int argc, char **argv)
{
    struct addrinfo *addr;
    struct rdma_cm_event *event = NULL;
    struct rdma_cm_id *conn= NULL;
    struct rdma_event_channel *ec = NULL;

    if (argc != 3)
        die("usage: client <server-address> <server-port>");

    TEST_NZ(getaddrinfo(argv[1], argv[2], NULL, &addr));

    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
    TEST_NZ(rdma_resolve_addr(conn, NULL, addr->ai_addr, TIMEOUT_IN_MS));

    freeaddrinfo(addr);

    while (0 == rdma_get_cm_event(ec, &event))
        //while (rdma_get_cm_event(ec, &event))
    {
        std::cout << "rdma_get_cm_event\n";

        struct rdma_cm_event event_copy;

        memcpy(&event_copy, event, sizeof(*event));
        rdma_ack_cm_event(event);

        if (on_event(&event_copy))
            break;
    }

    rdma_destroy_event_channel(ec);

    return 0;
}

void die(const char *reason)
{
    fprintf(stderr, "%s\n", reason);
    exit(EXIT_FAILURE);
}

void build_context(struct ibv_context *verbs)
{
    if (s_ctx)
    {
        if (s_ctx->ctx != verbs)
            die("cannot handle events in more than one context.");

        return;
    }

    s_ctx = (struct context *)malloc(sizeof(struct context));

    s_ctx->ctx = verbs;

    TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
    TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
    TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, 100, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));

    TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

void *SendMessages(void *context)
{
    static int loopcount = 0;
    while(1)
    {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck, message_available);
        //std::this_thread::sleep_for(std::chrono::microseconds(50));

        ok_to_send_next_message = 0;

        struct connection *conn = (struct connection *)context;
        struct ibv_send_wr wr, *bad_wr = NULL;
        struct ibv_sge sge;

        std::cout << "looping send..." << loopcount << '\n' << std::flush;

        memset(&wr, 0, sizeof(wr));

        wr.wr_id = (uintptr_t)conn;
        wr.opcode = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        sge.addr = (uintptr_t)conn->send_region;
        sge.length = BUFFER_SIZE;
        sge.lkey = conn->send_mr->lkey;

        snprintf(conn->send_region, BUFFER_SIZE, "message from active/client side with count %d", loopcount++);
        TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
    }
    return NULL;
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr)
{
    std::cout << "build_qp_attr\n";

    memset(qp_attr, 0, sizeof(*qp_attr));

    qp_attr->send_cq = s_ctx->cq;
    qp_attr->recv_cq = s_ctx->cq;
    qp_attr->qp_type = IBV_QPT_RC;

    qp_attr->cap.max_send_wr = 100;
    qp_attr->cap.max_recv_wr = 100;
    qp_attr->cap.max_send_sge = 1;
    qp_attr->cap.max_recv_sge = 1;
}

void * poll_cq(void *ctx)
{
    struct ibv_cq *cq;
    struct ibv_wc wc;

    while (1)
    {
        TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));
        ibv_ack_cq_events(cq, 1);
        TEST_NZ(ibv_req_notify_cq(cq, 0));

        int ne;
        struct ibv_wc wc;

        do
        {
            std::cout << "polling\n";
            ne = ibv_poll_cq(cq, 1, &wc);
        }
        while(ne == 0);

        on_completion(&wc);

        //if (wc.opcode == IBV_WC_SEND)
        if (wc.status == IBV_WC_SUCCESS)
        {
            {
                ok_to_send_next_message = 1;
                //while (message_available()) std::this_thread::yield();
                //std::cout << "past yield\n";
                std::unique_lock<std::mutex> lck(mtx);
                cv.notify_one();
            }
        }
    }

    return NULL;
}

void post_receives(struct connection *conn)
{
    std::cout << "post_receives\n";

    struct ibv_recv_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    wr.wr_id = (uintptr_t)conn;
    wr.next = NULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    sge.addr = (uintptr_t)conn->recv_region;
    sge.length = BUFFER_SIZE;
    sge.lkey = conn->recv_mr->lkey;

    TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void register_memory(struct connection *conn)
{
    std::cout << "register_memory\n";

    conn->send_region = (char *)malloc(BUFFER_SIZE);
    conn->recv_region = (char *)malloc(BUFFER_SIZE);

    TEST_Z(conn->send_mr = ibv_reg_mr(
                               s_ctx->pd,
                               conn->send_region,
                               BUFFER_SIZE,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

    TEST_Z(conn->recv_mr = ibv_reg_mr(
                               s_ctx->pd,
                               conn->recv_region,
                               BUFFER_SIZE,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
}

int on_addr_resolved(struct rdma_cm_id *id)
{
    std::cout << "on_addr_resolved\n";

    struct ibv_qp_init_attr qp_attr;
    struct connection *conn;

    build_context(id->verbs);
    build_qp_attr(&qp_attr);

    TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));

    id->context = conn = (struct connection *)malloc(sizeof(struct connection));

    conn->id = id;
    conn->qp = id->qp;
    conn->num_completions = 0;

    register_memory(conn);
    post_receives(conn);

    TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

    return 0;
}

void on_completion(struct ibv_wc *wc)
{
    std::cout << "on_completion\n";

    struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;

    if (wc->status != IBV_WC_SUCCESS)
    {
        //die("\ton_completion: status is not IBV_WC_SUCCESS.");
        printf("\ton_completion: status is not IBV_WC_SUCCESS.");
        printf("\t it is %d ", wc->status);
    }

    printf("\n");

    if (wc->opcode & IBV_WC_RECV)
        printf("\treceived message: %s\n", conn->recv_region);
    else if (wc->opcode == IBV_WC_SEND)
        printf("\tsend completed successfully.\n");
    else
        die("\ton_completion: completion isn't a send or a receive.");

    if (5 == ++conn->num_completions)
        rdma_disconnect(conn->id);
}

int on_connection(void *context)
{
    std::cout << "on_connection\n";

    TEST_NZ(pthread_create(&msgThread, NULL, SendMessages, context));

    return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
    struct connection *conn = (struct connection *)id->context;

    printf("disconnected.\n");

    rdma_destroy_qp(id);

    ibv_dereg_mr(conn->send_mr);
    ibv_dereg_mr(conn->recv_mr);

    free(conn->send_region);
    free(conn->recv_region);

    free(conn);

    rdma_destroy_id(id);

    return 1; /* exit event loop */
}

int on_route_resolved(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    printf("route resolved.\n");

    memset(&cm_params, 0, sizeof(cm_params));
    TEST_NZ(rdma_connect(id, &cm_params));

    return 0;
}

int on_event(struct rdma_cm_event *event)
{
    int r = 0;

    if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED)
        r = on_addr_resolved(event->id);
    else if (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED)
        r = on_route_resolved(event->id);
    else if (event->event == RDMA_CM_EVENT_ESTABLISHED)
        r = on_connection(event->id->context);
    else if (event->event == RDMA_CM_EVENT_DISCONNECTED)
        r = on_disconnect(event->id);
    else
        die("on_event: unknown event.");

    return r;
}
