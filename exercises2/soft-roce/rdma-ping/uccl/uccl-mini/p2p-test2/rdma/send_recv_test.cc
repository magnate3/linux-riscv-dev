/**
 * @file rdma_test.cc
 * @brief Test for UCCL RDMA transport
 */

#include "transport.h"
#include <cstdint>
#ifndef __HIP_PLATFORM_AMD__
//#include "cuda_runtime.h"
#else
#include "hip/hip_runtime.h"
#endif
#include "transport_config.h"
#include "util/util.h"
#include "util_timer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <signal.h>

// #define GPU 7

using namespace uccl;

static bool volatile quit = false;

int size =   0;
//size_t size =   0;
//FLAGS_msize * FLAGS_nreq * FLAGS_nmsg;
std::optional<RDMAEndpoint> ep;
struct {
  uint32_t port;
  uint32_t ip;
} local_p2p_listen;
struct {
  uint32_t port;
  uint32_t ip;
} remote_p2p_listen;

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_string(perftype, "tpt", "Performance type: basic/lat/tpt/bi.");
DEFINE_bool(warmup, false, "Whether to run warmup.");
DEFINE_uint32(nflow, 4, "Number of flows.");
DEFINE_uint32(nmsg, 1, "Number of messages within one request to post.");
DEFINE_uint32(nreq, 2, "Outstanding requests to post.");
DEFINE_uint32(msize, 1000000, "Size of message.");
DEFINE_uint32(iterations, 1000000, "Number of iterations to run.");
DEFINE_bool(flush, false, "Whether to flush after receiving.");
DEFINE_bool(bi, false, "Whether to run bidirectional test.");
DEFINE_uint32(oobport, 19999, "OOB port to use for bootstrapping.");

#if 0
bool ep_advertise(ConnID conn_id,  struct Mhandle* mhandle, void* addr,
                         size_t len, char* out_buf) {
  //auto* conn = conn_id_to_conn_[conn_id];
  //auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  //uccl::ucclRequest req_data;
  if (ep->prepare_fifo_metadata(
          static_cast<uccl::UcclFlow*>(conn_id.context), &mhandle,
          addr, len, out_buf) == -1)
    return false;
  return true;
}
#endif
static void server_basic(ConnID conn_id, struct Mhandle* mhandle, void* data, size_t size) {
  uccl::ucclRequest ureq;
  memset(&ureq, 0, sizeof(uccl::ucclRequest));
  ep->uccl_read_one((static_cast<uccl::UcclFlow*>(conn_id.context)), mhandle, data, size, &ureq);
#if 0
  int rc;
  int slot, nmsg;
  //uccl::FifoItem slot_item;
  //if (!flow->check_fifo_ready(&slot, &nmsg)) return -1;
  while(!flow->check_fifo_ready(&slot, &nmsg)) ;
  DCHECK(slot < kMaxReq && nmsg <= kMaxRecv) << slot << ", nmsg" << nmsg;
  auto send_comm = (static_cast<uccl::UcclFlow*>(conn_id.context))->send_comm_;
  auto ureqs = send_comm->fifo_ureqs[slot];
  auto rem_fifo = send_comm->base.fifo;
  volatile struct FifoItem* slots = rem_fifo->elems[slot];

  uccl::FifoItem slot_item = slots[0];
  do {
    rc = ep->uccl_read_async(
        static_cast<uccl::UcclFlow*>(conn_id.context), mhandle, data,
        size, slot_item, &ureq);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  while (!ep->uccl_poll_ureq_once(&ureq)) {
  }
#endif
}

static void client_basic(ConnID conn_id, struct Mhandle* mhandle, void* data) {
	//if(!ep_advertise(conn_id,mhandle))
    while (!quit) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
   }
}


uint64_t volatile tx_cur_sec_bytes = 0;
uint64_t tx_prev_sec_bytes = 0;
uint64_t volatile rx_cur_sec_bytes = 0;
uint64_t rx_prev_sec_bytes = 0;

uint64_t volatile c_itr = 0;
uint64_t volatile s_itr = 0;


static void server_worker(void) {
  uint64_t mr_id;
  std::string remote_ip;

  ConnID conn_id;
  struct Mhandle* mhandle;
  char buf[64] ={0};


  int remote_dev;
  conn_id = ep->test_uccl_accept(0, 0, remote_ip, &remote_dev);
  printf("Server accepted connection from %s \n", remote_ip.c_str());
#ifdef GPU
    void* data;
#ifndef __HIP_PLATFORM_AMD__
    cudaSetDevice(GPU);
    cudaMalloc(&data, size);
#else
    CHECK(hipSetDevice(GPU) == hipSuccess);
    CHECK(hipMalloc(&data, size) ==
          hipSuccess);
#endif
#else
    void* data =
        mmap(nullptr, size,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(data,0x42,size);
#endif
    assert(data != MAP_FAILED);
    ep->uccl_regmr((UcclFlow*)conn_id.context, data,
                  size, 0, &mhandle);
#if 0

    struct ucclRequest ureq;
    DCHECK(ep->uccl_recv_async((UcclFlow*)conn_id.context, &mhandle, &data,
                               &size, 1, &ureq) == 0);
    ep->uccl_poll_ureq(&ureq);
    memcpy(buf,data,63);
    printf("recv buf data : %s \n",buf);
#else
  uccl::ucclRequest ureq;
  memset(&ureq, 0, sizeof(uccl::ucclRequest));
  memcpy(buf,data,63);
  printf("buf data : %s \n",buf);
  if (ep->submit_fifo_metadata( static_cast<uccl::UcclFlow*>(conn_id.context), &mhandle, data, size, &ureq) == -1)
  {
        goto out;
  }
  printf("submit_fifo_metadata successfully \n");
  ep->uccl_poll_ureq(&ureq);
  memcpy(buf,data,63);
  printf("recv buf data : %s \n",buf);
#endif
  while (!quit) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
out:
  ep->uccl_deregmr(mhandle);
  munmap(data, size);
}


static void client_worker(void) {
 ConnID conn_id;
 void* datas;
 struct Mhandle* mhandle;
 char out_buf[64] = {0};
 char buf[64] = {0};
    std::string dataplane_remote_ip = ip_to_str(remote_p2p_listen.ip);
    int dataplane_remote_port = remote_p2p_listen.port;
    printf("Client connecting to %s:%d \n", dataplane_remote_ip.c_str(), dataplane_remote_port);
    conn_id = ep->test_uccl_connect(0, 0, 0, 0, dataplane_remote_ip,
                                         dataplane_remote_port);
    printf("Client connected to %s:%d \n", dataplane_remote_ip.c_str(), dataplane_remote_port);
#ifdef GPU
    void* data;
#ifndef __HIP_PLATFORM_AMD__
    cudaSetDevice(GPU);
    cudaMalloc(&data, size);
#else
    CHECK(hipSetDevice(GPU) == hipSuccess);
    CHECK(hipMalloc(&data, size) ==
          hipSuccess);
#endif
#else
    void* data =
        mmap(nullptr, size,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(data,0x41,size);
#endif
    assert(data != MAP_FAILED);
    ep->uccl_regmr((UcclFlow*)conn_id.context, data, size, 0, &mhandle);


#if 0
  printf("prepare_fifo_metadata \n");
  uccl::ucclRequest ureq;
  memset(&ureq, 0, sizeof(uccl::ucclRequest));
  //if (ep->prepare_fifo_metadata( static_cast<uccl::UcclFlow*>(conn_id.context), &mhandle, data, size, out_buf) == -1)
  if (ep->submit_fifo_metadata( static_cast<uccl::UcclFlow*>(conn_id.context), &mhandle, data, size, &ureq) == -1)
  {
        goto out;
  }
  printf("prepare_fifo_metadata successfully \n");
  client_basic(conn_id, mhandle, data);
#else
      memcpy(buf,data,63);
      printf("send buf data : %s \n",buf);
      uccl::ucclRequest ureq;
      memset(&ureq, 0, sizeof(uccl::ucclRequest));
      while (ep->uccl_send_async((UcclFlow*)conn_id.context, mhandle, data,
                                   size, &ureq)) {
      }
      ep->uccl_poll_ureq(&ureq);
#endif
out:
  ep->uccl_deregmr(mhandle);
  munmap(data, size);
}

// clang-format off
// TO run on AMD cluster:
// export NCCL_IB_HCA="rdma0:1,rdma2:1,rdma3:1,rdma4:1"
// export HIP_VISIBLE_DEVICES=1,2,0,5
// LD_LIBRARY_PATH="${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH}" ./transport_test --server=true
// LD_LIBRARY_PATH="${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH}" ./transport_test --serverip=10.42.19.1
// clang-format on

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  size =   FLAGS_msize * FLAGS_nreq * FLAGS_nmsg;
  ep.emplace(ucclParamNUM_ENGINES());
  ep->initialize_engine_by_dev(0, true);
  local_p2p_listen.port = ep->get_p2p_listen_port(0);
  local_p2p_listen.ip = str_to_ip(ep->get_p2p_listen_ip(0));

  if (FLAGS_server) {
    listen_accept_exchange(FLAGS_oobport, &local_p2p_listen,
                           sizeof(local_p2p_listen), &remote_p2p_listen,
                           sizeof(remote_p2p_listen));
    server_worker();
  } else {
    CHECK(!FLAGS_serverip.empty()) << "Server IP address is required.";
    connect_exchange(FLAGS_oobport, FLAGS_serverip, &local_p2p_listen,
                     sizeof(local_p2p_listen), &remote_p2p_listen,
                     sizeof(remote_p2p_listen));
    client_worker();
  }

  return 0;
}
