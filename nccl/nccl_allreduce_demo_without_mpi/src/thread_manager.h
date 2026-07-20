#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "nccl_helper.h"

class Barrier {
 private:
  std::mutex mutex;
  std::condition_variable cv;
  std::size_t count;
  std::size_t threshold;
  std::size_t generation;

 public:
  explicit Barrier(std::size_t count) : count(count), threshold(count), generation(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    std::size_t gen = generation;

    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
      return;
    }

    cv.wait(lock, [this, gen] { return gen != generation; });
  }
};

struct GPUThreadContext {
  int deviceId;
  int globalRank;
  int worldSize;
  size_t dataSize;
  ncclComm_t comm;
  cudaStream_t stream;
  float* d_data;
  float* d_result;
  bool success;
  int nodeRank;

  double avgTimeMs;
  double bandwidth;

  GPUThreadContext()
      : deviceId(-1),
        globalRank(-1),
        worldSize(0),
        dataSize(0),
        comm(nullptr),
        stream(nullptr),
        d_data(nullptr),
        d_result(nullptr),
        success(false),
        nodeRank(0),
        avgTimeMs(0),
        bandwidth(0) {}
};

void gpuWorkerThread(GPUThreadContext& ctx, const ncclUniqueId& ncclId, Barrier& initBarrier, Barrier& syncBarrier,
                     std::atomic<bool>& initError, int iterations) {
  Logger logger(ctx.nodeRank);

  try {
    CUDA_CHECK(cudaSetDevice(ctx.deviceId));

    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    NCCLHelper::printDeviceInfo(ctx.deviceId, ctx.globalRank);

    CUDA_CHECK(cudaMalloc(&ctx.d_data, ctx.dataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_result, ctx.dataSize * sizeof(float)));

    std::vector<float> h_data(ctx.dataSize, ctx.globalRank + 1.0f);
    CUDA_CHECK(cudaMemcpy(ctx.d_data, h_data.data(), ctx.dataSize * sizeof(float), cudaMemcpyHostToDevice));

    initBarrier.wait();

    logger.nodeLog("GPU ", ctx.deviceId, " initializing NCCL communicator with rank ", ctx.globalRank, " / ",
                   ctx.worldSize);

    NCCL_CHECK(ncclCommInitRank(&ctx.comm, ctx.worldSize, ncclId, ctx.globalRank));
    logger.nodeLog("GPU ", ctx.deviceId, " (global rank ", ctx.globalRank, ") initialized NCCL communicator");

    syncBarrier.wait();

    NCCL_CHECK(ncclAllReduce(ctx.d_data, ctx.d_result, ctx.dataSize, ncclFloat, ncclSum, ctx.comm, ctx.stream));

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    std::vector<float> h_result(ctx.dataSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), ctx.d_result, ctx.dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    float expectedSum = ctx.worldSize * (ctx.worldSize + 1) / 2.0f;
    bool correct = true;

    size_t checkLimit = std::min(ctx.dataSize, static_cast<size_t>(10));
    for (size_t i = 0; i < checkLimit; i++) {
      if (std::abs(h_result[i] - expectedSum) > 1e-5) {
                logger.nodeLog("GPU ", ctx.deviceId, " (global rank ", ctx.globalRank, 
                             "): Validation failed at index ", i,
                             ", got ", h_result[i], ", expected ", expectedSum