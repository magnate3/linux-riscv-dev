#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0) // IF zero, return Error

const int BUFFER_SIZE = 1024;

struct context {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_comp_channel *comp_channel;

  pthread_t cq_poller_thread;
};

struct connection {
  struct ibv_qp *qp;

  struct ibv_mr *recv_mr;
  struct ibv_mr *send_mr;

  char *recv_region;
  char *send_region;
};

static void die(const char *reason);

static void build_context(struct ibv_context *verbs);
static void build_qp_attr(struct ibv_qp_init_attr *qp_attr);
static void * poll_cq(void *);
static void post_receives(struct connection *conn);
static void register_memory(struct connection *conn);

static void on_completion(struct ibv_wc *wc);
static int on_connect_request(struct rdma_cm_id *id);
static int on_connection(void *context);
static int on_disconnect(struct rdma_cm_id *id);
static int on_event(struct rdma_cm_event *event);

static struct context *s_ctx = NULL;

/* Questions 
1. What's the difference between event channel & event id?
2. What's
*/

int main(int argc, char **argv)
{
  struct sockaddr_in6 addr;             // Socket server address?
  struct rdma_cm_event *event = NULL;   // RDMA event, There are 3 basic events: connect request, connected, disconnected
  struct rdma_cm_id *listener = NULL;   // RDMA id
  struct rdma_event_channel *ec = NULL; // RDMA event channel
  uint16_t port = 0;                    // Socket server port?

  memset(&addr, 0, sizeof(addr));       // Clear address
  addr.sin6_family = AF_INET6;          // USE TCP

  TEST_Z(ec = rdma_create_event_channel()); // Create RDMA event channel: ec, not connect to server yet
  TEST_NZ(rdma_create_id(ec, &listener, NULL, RDMA_PS_TCP)); // Use event channel: [ec] to create rdma_id: [listener]
  TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr)); // Connect to server, do something to rdma_id: [listenser]
  TEST_NZ(rdma_listen(listener, 10)); /* backlog=10 is arbitrary */ // Start to listen, by rdma_id: [listener]

  port = ntohs(rdma_get_src_port(listener)); // Get client binding port

  printf("listening on port %d.\n", port);

  while (rdma_get_cm_event(ec, &event) == 0) { // Get next event from event channel: [ec], save to struct event
    struct rdma_cm_event event_copy;

    memcpy(&event_copy, event, sizeof(*event));
    rdma_ack_cm_event(event);  // rdma client reply that had getting event

    if (on_event(&event_copy)) // Deal with getting event, if return true, meaning end of the transfering
      break;
  }

  rdma_destroy_id(listener);   // Destory rdma id: [listener]
  rdma_destroy_event_channel(ec); // Desctory rdma channel: [ec]

  return 0;
}

void die(const char *reason)
{
  fprintf(stderr, "%s\n", reason);
  exit(EXIT_FAILURE);
}

void build_context(struct ibv_context *verbs)
{
  if (s_ctx) {
    if (s_ctx->ctx != verbs)
      die("cannot handle events in more than one context.");

    return;
  }

  s_ctx = (struct context *)malloc(sizeof(struct context));

  s_ctx->ctx = verbs;

  TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
  TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
  TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, 10, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */
  TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));

  TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, NULL));
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr)
{
  memset(qp_attr, 0, sizeof(*qp_attr));

  qp_attr->send_cq = s_ctx->cq;
  qp_attr->recv_cq = s_ctx->cq;
  qp_attr->qp_type = IBV_QPT_RC;

  qp_attr->cap.max_send_wr = 10;
  qp_attr->cap.max_recv_wr = 10;
  qp_attr->cap.max_send_sge = 1;
  qp_attr->cap.max_recv_sge = 1;
}

void * poll_cq(void *ctx)
{
  struct ibv_cq *cq;
  struct ibv_wc wc;

  while (1) {
    TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));
    ibv_ack_cq_events(cq, 1);
    TEST_NZ(ibv_req_notify_cq(cq, 0));

    while (ibv_poll_cq(cq, 1, &wc))
      on_completion(&wc);
  }

  return NULL;
}

void post_receives(struct connection *conn)
{
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = (uintptr_t)conn;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)conn->recv_region;
  sge.length = BUFFER_SIZE;
  sge.lkey = conn->recv_mr->lkey;

  TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr)); // ibv key point
}

void register_memory(struct connection *conn)
{
  conn->send_region = malloc(BUFFER_SIZE); // malloc BUFFER_SIZE memory space
  conn->recv_region = malloc(BUFFER_SIZE);

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

void on_completion(struct ibv_wc *wc)
{
  if (wc->status != IBV_WC_SUCCESS)
    die("on_completion: status is not IBV_WC_SUCCESS.");

  if (wc->opcode & IBV_WC_RECV) {
    struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;

    printf("received message: %s\n", conn->recv_region);

  } else if (wc->opcode == IBV_WC_SEND) {
    printf("send completed successfully.\n");
  }
}

/*
 * Client dealing connect request event
 * id: ? server's id
 */
int on_connect_request(struct rdma_cm_id *id)
{
  struct ibv_qp_init_attr qp_attr;
  struct rdma_conn_param cm_params;
  struct connection *conn;

  printf("received connection request.\n");

  build_context(id->verbs); // TODO: client side builds it's local context, not beloging to RDMA library
  build_qp_attr(&qp_attr);  // TODO: client side builds queue?

  TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));

  id->context = conn = (struct connection *)malloc(sizeof(struct connection));
  conn->qp = id->qp;

  register_memory(conn); // TODO: register memory
  post_receives(conn);   // TODO: 

  memset(&cm_params, 0, sizeof(cm_params));
  TEST_NZ(rdma_accept(id, &cm_params));

  return 0;
}

int on_connection(void *context)
{
  struct connection *conn = (struct connection *)context;
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  snprintf(conn->send_region, BUFFER_SIZE, "message from passive/server side with pid %d", getpid());

  printf("connected. posting send...\n");

  memset(&wr, 0, sizeof(wr));

  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  sge.addr = (uintptr_t)conn->send_region;
  sge.length = BUFFER_SIZE;
  sge.lkey = conn->send_mr->lkey;

  TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));

  return 0;
}

int on_disconnect(struct rdma_cm_id *id)
{
  struct connection *conn = (struct connection *)id->context;

  printf("peer disconnected.\n");

  rdma_destroy_qp(id);

  ibv_dereg_mr(conn->send_mr);
  ibv_dereg_mr(conn->recv_mr);

  free(conn->send_region);
  free(conn->recv_region);

  free(conn);

  rdma_destroy_id(id);

  return 0;
}

int on_event(struct rdma_cm_event *event)
{
  int r = 0;

  if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST) // Client got RDMA connect request? why client reciving request, ---> according to csdi project pdf, client send this event
    r = on_connect_request(event->id);
  else if (event->event == RDMA_CM_EVENT_ESTABLISHED) // Client got RDMA connect established, ---> client receive this event
    r = on_connection(event->id->context);
  else if (event->event == RDMA_CM_EVENT_DISCONNECTED) // Client got RDMA connect disconnected
    r = on_disconnect(event->id);
  else
    die("on_event: unknown event."); // Client got unknown event

  return r;
}