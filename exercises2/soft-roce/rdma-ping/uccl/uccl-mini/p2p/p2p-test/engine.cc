#include "engine.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

int const kMaxNumGPUs = 1;
// Assume the local and remote GPUs have the same GPU-NIC mapping.
uint8_t gpu_to_dev[kMaxNumGPUs] = {0};

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  py::gil_scoped_release release;
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;

  google::InitGoogleLogging("uccl_p2p");
  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(ucclParamNUM_ENGINES());

  auto gpu_cards = uccl::get_gpu_cards();
  std::cout << "Creating Engine GPU num: "<< gpu_cards.size() <<  std::endl;
#ifndef CPU_MEMORY
  DCHECK(local_gpu_idx_ < gpu_cards.size() && gpu_cards.size() <= kMaxNumGPUs)
      << "Local GPU index out of range";
  auto ib_nics = uccl::get_rdma_nics();
  // Find the RDMA NIC that is closest to each of the GPUs.
  for (int i = 0; i < kMaxNumGPUs; i++) {
    auto gpu_device_path = gpu_cards[i];
    auto ib_nic_it = std::min_element(
        ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
          return uccl::cal_pcie_distance(gpu_device_path, a.second) <
                 uccl::cal_pcie_distance(gpu_device_path, b.second);
        });
    gpu_to_dev[i] = ib_nic_it - ib_nics.begin();
  }
  std::cout << "Detected best GPU-NIC mapping: " << std::endl;
  for (int i = 0; i < kMaxNumGPUs; i++) {
    std::cout << "\tGPU " << i << " -> NIC " << gpu_to_dev[i] << " ("
              << ib_nics[gpu_to_dev[i]].first << ")" << std::endl;
  }
  std::cout << std::endl;
#endif
  // Initialize the engine based on the GPU index.
#ifdef LAZY_CREATE_ENGINE
  ep_->initialize_engine_by_dev(gpu_to_dev[local_gpu_idx_]);
#endif

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  py::gil_scoped_release release;
  std::cout << "Destroying Engine..." << std::endl;
  delete ep_;

  for (auto& [conn_id, conn] : conn_id_to_conn_) {
    delete conn;
  }
  for (auto& [mr_id, mr] : mr_id_to_mr_) {
    delete mr;
  }

  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string const& ip_addr, int const& remote_gpu_idx,
                       uint64_t& conn_id) {
  py::gil_scoped_release release;
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << std::endl;

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id = ep_->test_uccl_connect(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, gpu_to_dev[remote_gpu_idx],
      remote_gpu_idx, ip_addr);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  py::gil_scoped_release release;
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  uccl::ConnID uccl_conn_id = ep_->test_uccl_accept(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, ip_addr, &remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  py::gil_scoped_release release;

  mr_id = next_mr_id_.fetch_add(1);

  uccl::Mhandle* mhandle;
  ep_->uccl_regmr(gpu_to_dev[local_gpu_idx_], const_cast<void*>(data), size, 0,
                  &mhandle);

  mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};

  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  py::gil_scoped_release release;
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  int rc;
  do {
    rc = ep_->uccl_send_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
        data, size, &ureq);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq);

  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data,
                    size_t max_size, size_t& recv_size) {
  py::gil_scoped_release release;
  uccl::ucclRequest ureq;

  auto conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  int max_size_int = static_cast<int>(max_size);

  int rc;
  do {
    rc = ep_->uccl_recv_async(
        static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
        &data, &max_size_int, 1, &ureq);
    if (rc == -1) {
      std::this_thread::yield();
    }
  } while (rc == -1);

  ep_->uccl_poll_ureq(&ureq);

  recv_size = ureq.recv.data_len[0];
  return true;
}
