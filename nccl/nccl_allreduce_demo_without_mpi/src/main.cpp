#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_worker.h"
#include "tcp_socket.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  try {
    AppArgs args = AppArgs::parseArgs(argc, argv);
    args.printConfig();

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    Logger::log("Running on host:", hostname, "as node", args.nodeRank);

    ncclUniqueId ncclId;
    if (args.nodeRank == 0) {
      NCCL_CHECK(ncclGetUniqueId(&ncclId));
      Logger::log("Generated NCCL Unique ID");
    }

    if (!NCCLIdBroadcaster::broadcastNCCLId(ncclId, args.nodeRank, args.nodeCount, args.masterIP, args.port)) {
      throw std::runtime_error("Failed to broadcast NCCL ID");
    }

    GPUWorkerManager gpuManager(args.nodeRank, args.nodeCount, args.dataSize, args.iterations);

    int localDeviceCount = gpuManager.getLocalDeviceCount();
    Logger::log("Node", args.nodeRank, "managing", localDeviceCount, "GPUs");

    if (!gpuManager.runAllreduce(ncclId)) {
      throw std::runtime_error("AllReduce operation failed");
    }

    double avgTime, avgBandwidth;
    gpuManager.getPerformanceStats(avgTime, avgBandwidth);

    Logger::log("\nNode", args.nodeRank, "Performance Summary:");
    Logger::log("  Data size:", args.dataSize, "elements (", (args.dataSize * sizeof(float) / (1024.0 * 1024.0)),
                "MB)");
    Logger::log("  Average time:", std::fixed, std::setprecision(3), avgTime, "ms");
    Logger::log("  Bandwidth:", std::fixed, std::setprecision(2), avgBandwidth, "GB/s");

    if (args.nodeRank == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      Logger::log("\nAll operations completed successfully across", args.nodeCount, "nodes");
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}