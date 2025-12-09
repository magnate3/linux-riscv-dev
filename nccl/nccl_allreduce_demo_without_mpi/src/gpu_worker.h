#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <thread>
#include <vector>
#include "utils.h"

struct GPUContext {
  int deviceId;
  int localRank;
  int globalRank;
  int nodeRank;
  int localSize;
  int globalSize;
  size_t dataSize;
  ncclComm_t comm;
  cudaStream_t stream;
  float* d_input;
  float* d_output;
  bool success;

  double avgTimeMs;
  double bandwidth;

  GPUContext()
      : deviceId(-1),
        localRank(-1),
        globalRank(-1),
        nodeRank(-1),
        localSize(0),
        globalSize(0),
        dataSize(0),
        comm(nullptr),
        stream(nullptr),
        d_input(nullptr),
        d_output(nullptr),
        success(false),
        avgTimeMs(0.0),
        bandwidth(0.0) {}
};

class GPUWorkerManager {
 public:
  GPUWorkerManager(int nodeRank, int nodeCount, size_t dataSize, int iterations);

  ~GPUWorkerManager();

  bool runAllreduce(const ncclUniqueId& ncclId);

  std::vector<GPUContext>& getContexts() { return contexts; }

  bool isSuccessful() const { return !initError; }

  int getLocalDeviceCount() const { return deviceCount; }

  void getPerformanceStats(double& avgTime, double& avgBandwidth) const;

 private:
  int nodeRank;
  int nodeCount;
  int deviceCount;
  size_t dataSize;
  int iterations;

  std::vector<std::thread> threads;
  std::vector<GPUContext> contexts;
  std::atomic<bool> initError;

  std::unique_ptr<Barrier> initBarrier;
  std::unique_ptr<Barrier> syncBarrier;

  static void gpuWorkerThread(GPUContext& ctx, Barrier& initBarrier, Barrier& syncBarrier, std::atomic<bool>& initError,
                              int iterations, const ncclUniqueId& ncclId);
};