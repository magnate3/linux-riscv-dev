#pragma once

#include "param.h"
#include <cstdint>
#include <string>
#include <thread>
#include <unistd.h>

// #define STATS
#ifndef LAZY_CREATE_ENGINE
#define LAZY_CREATE_ENGINE
#endif

/// Interface configuration.
// For Azure HPC Ubuntu 22.04 only, normally it should be "mlx5_".
static char const* IB_DEVICE_NAME_PREFIX = "mlx5_";
//static char const* IB_DEVICE_NAME_PREFIX = "mlx5_ib";
static constexpr bool ROCE_NET = true;
// If SINGLE_CTRL_NIC is set, all devices will use the same IP.
static std::string SINGLE_CTRL_NIC("enp61s0f1np1");
static constexpr uint8_t DEVNAME_SUFFIX_LIST[8] = {0, 1, 0, 0, 0, 0, 0, 0};
static constexpr uint8_t NUM_DEVICES = 1;
static constexpr double LINK_BANDWIDTH = 100.0 * 1e9 / 8;  // 1m00Gbps
//static constexpr double LINK_BANDWIDTH = 200.0 * 1e9 / 8;  // 200Gbps

// Whether to pin the thread to the NUMA node.
UCCL_PARAM(PIN_TO_NUMA, "PIN_TO_NUMA", 1);
// Traffic class for RoCE.
UCCL_PARAM(ROCE_TRAFFIC_CLASS, "ROCE_TRAFFIC_CLASS", 3);
// Service level for RoCE.
UCCL_PARAM(ROCE_SERVICE_LEVEL, "ROCE_SERVICE_LEVEL", 135);
// GID index for RoCE.
UCCL_PARAM(ROCE_GID_IDX, "ROCE_GID_IDX", 3);
// Service level for IB.
UCCL_PARAM(IB_SERVICE_LEVEL, "IB_SERVICE_LEVEL", 0);
// GID index for IB.
UCCL_PARAM(IB_GID_IDX, "IB_GID_IDX", 0);

#ifdef BROADCOM_NIC
UCCL_PARAM(RCMode, "RCMODE", true);
#else
// Use RC for data transfer.
//UCCL_PARAM(RCMode, "RCMODE", false);
UCCL_PARAM(RCMode, "RCMODE", true);
#endif

// Bypass the pacing stage.
UCCL_PARAM(BypassPacing, "BYPASS_PACING", true);

#ifndef __HIP_PLATFORM_AMD__
// # of engines per device.
UCCL_PARAM(NUM_ENGINES, "NUM_ENGINES", 4);
// Path/QP per engine.
UCCL_PARAM(PORT_ENTROPY, "PORT_ENTROPY", 32);
// Maximum chunk size for each WQE.
UCCL_PARAM(CHUNK_SIZE_KB, "CHUNK_SIZE_KB", 64);
#else
UCCL_PARAM(NUM_ENGINES, "NUM_ENGINES", 1);
UCCL_PARAM(PORT_ENTROPY, "PORT_ENTROPY", 256);
UCCL_PARAM(CHUNK_SIZE_KB, "CHUNK_SIZE_KB", 128);
#endif

// Broadcom NICs do not support ibv_cq_ex.
#ifndef BROADCOM_NIC
#define USE_CQ_EX
#endif

static constexpr uint32_t MAX_PEER = 256;
// Maximum number of flows (one-way) on each engine.
static constexpr uint32_t MAX_FLOW = 256;

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Each dev use [ENGINE_CPU_START_LIST[dev], ENGINE_CPU_START_LIST[dev] +
// NUM_ENGINES)
static int64_t ENGINE_CPU_START_LIST[8] = {
    16,
    16 + ucclParamNUM_ENGINES(),
    16 + 2 * ucclParamNUM_ENGINES(),
    16 + 3 * ucclParamNUM_ENGINES(),
    96,
    96 + ucclParamNUM_ENGINES(),
    96 + 2 * ucclParamNUM_ENGINES(),
    96 + 3 * ucclParamNUM_ENGINES(),
};

static constexpr uint32_t kMaxAckWRs = 8;
static constexpr uint32_t UD_ADDITION = 40;
static constexpr uint32_t kMaxCtrlWRs = 2048;

// Limit the per-flow outstanding bytes on each engine.
static uint32_t kMaxUnAckedBytesPerFlow =
    2 * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
// Limit the outstanding bytes on each engine.
// Low means if a flow exceeds its own budget but doesn't exceed the Low
// threshold, it can send until Low threshold.
static uint32_t kMaxUnAckedBytesPerEngineLowForRoCE =
    (8) * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
static uint32_t kMaxUnAckedBytesPerEngineLowForIB =
    (18) * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
// High means if all flows reach this threshold, all flows can't send any bytes.
static uint32_t kMaxUnAckedBytesPerEngineHighForRoCE =
    (12) * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
static uint32_t kMaxUnAckedBytesPerEngineHighForIB =
    (24) * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
// Congestion control algorithm.
enum SenderCCA {
  SENDER_CCA_NONE,
  // Timely [SIGCOMM'15]
  SENDER_CCA_TIMELY,
  // Swift [SIGCOMM'20]
  SENDER_CCA_SWIFT,
};
enum ReceiverCCA {
  RECEIVER_CCA_NONE,
  // EQDS [NSDI'22]
  RECEIVER_CCA_EQDS,
};
static constexpr enum SenderCCA kSenderCCA = SENDER_CCA_TIMELY;
static constexpr enum ReceiverCCA kReceiverCCA = RECEIVER_CCA_NONE;
static_assert(kSenderCCA != SENDER_CCA_NONE ||
                  kReceiverCCA != RECEIVER_CCA_NONE,
              "At least one of the sender and receiver must have a congestion "
              "control algorithm.");

// Note that load-based policy shoud >= ENGINE_POLICY_LOAD.
enum engine_lb_policy {
  // Bind each flow to one engine.
  ENGINE_POLICY_BIND,
  // Round-robin among engines.
  ENGINE_POLICY_RR,
  // Choose obliviously.
  ENGINE_POLICY_OBLIVIOUS,
  // Load balancing based on the load of each engine.
  ENGINE_POLICY_LOAD,
  // Variant of ENGINE_POLICY_LOAD, which uses power of two.
  ENGINE_POLICY_LOAD_POT,
};
static constexpr enum engine_lb_policy kEngineLBPolicy = ENGINE_POLICY_RR;

static uint32_t const PACER_CPU_START = 3 * NUM_CPUS / 4;

static int const kTotalQP =
    ucclParamPORT_ENTROPY() +
    (kReceiverCCA == RECEIVER_CCA_EQDS ? 1 : 0) /* Credit QP */;
// Recv buffer size smaller than kRCSize will be handled by RC directly.
// static constexpr uint32_t kRCSize = 8192;
static constexpr uint32_t kRCSize = 0;
// fallback to nccl
// static constexpr uint32_t kRCSize = 4000000;
// Minimum post receive size in NCCL.
static constexpr uint32_t NCCL_MIN_POST_RECV = 65536;
// fallback to nccl
// static constexpr uint32_t NCCL_MIN_POST_RECV = 4000000;

// Limit the bytes of consecutive cached QP uses.
static constexpr uint32_t kMAXConsecutiveSameChoiceBytes = 16384;
// Message size threshold for allowing using cached QP.
static constexpr uint32_t kMAXUseCacheQPSize = 8192;
// Message size threshold for bypassing the timing wheel.
static constexpr uint32_t kBypassTimingWheelThres = 9000;

// DupAckThres equals to 1 means all duplicate acks are caused by
// packet loss. This is true for flow-level ECMP, which is the common case. When
// the network supports adaptive routing, duplicate acks may be caused by
// adaptive routing. In this case, DupAckThres should be set to a
// value greater than 0.
static constexpr uint32_t ROCE_DUP_ACK_THRES = 32;

static uint32_t kRetrChunkSize =
    (ucclParamCHUNK_SIZE_KB() << 10) + 12 /* sizeof(retr_chunk_hdr) */;

// # of Tx work handled in one loop.
static constexpr uint32_t kMaxTxWork = 4;
// Maximum number of Tx bytes to be transmitted in one loop.
static uint32_t kMaxTxBytesThres =
    32 * std::max((uint32_t)(ucclParamCHUNK_SIZE_KB() << 10), 32768u);
// # of Rx work handled in one loop.
static constexpr uint32_t kMaxRxWork = 8;
// Completion queue (CQ) size.
static constexpr int kCQSize = 16384;
// Interval for posting a signal WQE.
// static constexpr uint32_t kSignalInterval = kCQSize >> 1;
static constexpr uint32_t kSignalInterval = 1;
// Interval for syncing the clock with NIC.
static constexpr uint32_t kSyncClockIntervalNS = 100000;
// Maximum number of CQEs to retrieve in one loop.
static constexpr uint32_t kMaxBatchCQ = 16;
// CQ moderation count.
static constexpr uint32_t kCQMODCount = 32;
// CQ moderation period in microsecond.
static constexpr uint32_t kCQMODPeriod = 100;
// Maximum size of inline data.
static constexpr uint32_t kMaxInline = 128;
// Maximum number of SGEs in one WQE.
static constexpr uint32_t kMaxSge = 2;
// Maximum number of outstanding receive messages in one recv request.
static constexpr uint32_t kMaxRecv = 1;
// Maximum number of outstanding receive requests in one engine.
static constexpr uint32_t kMaxReq = 128;
// Maximum number of WQEs in SRQ (Shared Receive Queue).
static constexpr uint32_t kMaxSRQ = 16 * kMaxReq;
// Maximum number of chunks can be transmitted from timing wheel in one loop.
static constexpr uint32_t kMaxBurstTW = 8;
// Posting recv WQEs every kPostRQThreshold.
static constexpr uint32_t kPostRQThreshold = kMaxBatchCQ;
// When CQEs from one QP reach kMAXCumWQE, send immediate ack.
// 1 means always send immediate ack.
static constexpr uint32_t kMAXCumWQE = 4;
// When the cumulative bytes reach kMAXCumBytes, send immediate ack.
static uint32_t kMAXCumBytes = kMAXCumWQE * (ucclParamCHUNK_SIZE_KB() << 10);
// Before reaching it, the receiver will not consider that it has encountered
// OOO, and thus there is no immediate ack. This is to tolerate the OOO caused
// by the sender's qp scheduling.
static constexpr uint32_t kMAXRXOOO = 8;

// Sack bitmap size in bits.
// Note that kSackBitmapSize must be <= half the maximum value of UINT_CSN.
// E.g., UINT_CSN = 8bit, kSacBitmapSize <= 128.
static constexpr std::size_t kSackBitmapSize = 64 << 1;

// Maximum number of Retransmission Timeout (RTO) before aborting the flow.
static constexpr uint32_t kRTOAbortThreshold = 50;

static constexpr uint32_t kMAXRTTUS = 10000;
// Constant/Dynamic RTO.
static constexpr bool kConstRTO = true;
// kConstRTO == true: Constant retransmission timeout in microseconds.
static constexpr double kRTOUSec = 1000;
// kConstRTO == false: Minimum retransmission timeout in microseconds.
static constexpr double kMinRTOUsec = 1000;
static constexpr uint32_t kRTORTT = 4;  // RTO = kRTORTT RTTs

// Slow timer (periodic processing) interval in microseconds.
static constexpr size_t kSlowTimerIntervalUs = 1000;

/// Debugging and testing.
// Disable hardware timestamp.
static constexpr bool kTestNoHWTimestamp = false;
// Use constant(maximum) rate for transmission.
static constexpr bool kTestConstantRate = false;
// Test lossy network.
static constexpr bool kTestLoss = false;
static constexpr double kTestLossRate = 0.0;
