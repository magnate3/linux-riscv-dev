#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <THC/THC.h>

#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "../c10d/ProcessGroupNCCL.hpp"
#include "../c10d/TCPStore.hpp"
#include "../c10d/PrefixStore.hpp"

#include "../readerwriterqueue/readerwriterqueue.h"

using namespace c10d;
using namespace moodycamel;

// Hook on AccumulateGrad by default
#undef DIST_OPT_HOOK_TENSOR

// Store various *constant* options passed or calculated during initialization
struct DistributedOptimizerOptions {
  // Passed during initialization
  TORCH_ARG(bool, bias_correction) = true;
  TORCH_ARG(int, eps_mode) = 1; // eps_inside_sqrt = False
  // TODO: keep max_grad_norm not per parameter group for now
  TORCH_ARG(double, max_grad_norm) = 0;
  TORCH_ARG(bool, use_mt) = false;
  TORCH_ARG(double, amp_scale_adjustment) = 1;
  TORCH_ARG(bool, overlap_reductions) = true;
  TORCH_ARG(bool, full_pipeline) = true;
  TORCH_ARG(bool, compute_L2_grad_norm) = false;
  TORCH_ARG(long, num_blocks) = 4;
  TORCH_ARG(long, num_chunks) = 4;
  TORCH_ARG(long, num_rs_pg) = 1;
  TORCH_ARG(long, num_ar_pg) = 4;
  TORCH_ARG(long, num_ag_pg) = 0;
  TORCH_ARG(bool, exp_enabled) = false;
  TORCH_ARG(long, exp_num_rs_pg) = 1;
  TORCH_ARG(long, exp_num_ar_pg) = 4;
  TORCH_ARG(bool, flat_mt) = false;
  TORCH_ARG(bool, predivide) = true;
  TORCH_ARG(bool, e5m2_allgather) = false;
  TORCH_ARG(bool, do_not_flatten_model) = false;

  // Logging to stdout controlled by env DIST_OPT_LOGG
  TORCH_ARG(bool, logging) = false;

  // Calculated
  TORCH_ARG(int, world_size);
  TORCH_ARG(int, rank);
  TORCH_ARG(std::string, master_addr);
  TORCH_ARG(int, master_port);
  TORCH_ARG(int, device);
  TORCH_ARG(long, group_size) = 0;
  TORCH_ARG(long, num_groups);
  TORCH_ARG(long, group_rank);
  TORCH_ARG(long, rank_in_group);
  TORCH_ARG(long, net_total_param_size);
  TORCH_ARG(long, block_size);
  TORCH_ARG(long, chunk_size);
  TORCH_ARG(long, shard_size);
};

/** We only need step for now. */
struct DistributedFusedAdamParamState : public 
    torch::optim::OptimizerCloneableParamState<DistributedFusedAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;

public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  ~DistributedFusedAdamParamState() = default;
};

class AccGradPostHook;

/* Before initializing this, should call these API at Python side, and make sure
 * environment variables WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
 * are set correctly:
 *
 *   import torch
 *   import apex, amp_C
 *
 *   torch.distributed.init_process_group(backend='nccl', init_method="env://")
 *   assert torch.distributed.is_initialized()
 *   world_size = torch.distributed.get_world_size()
 *   rank = torch.distributed.get_rank()
 *   torch.cuda.set_device(rank)
 */
class DistributedFusedAdam : public torch::optim::Adam {
  public:
    DistributedFusedAdam(
          /** Since we only assume single param group for now,
           *  let's not use OptimizerParamGroup.
           */
          const std::vector<torch::Tensor> &_params,
          /** DistributedFusedAdamOptions, leave them here to keep
           *  them shown in Python front-end help.
           */
          double _lr,
          std::tuple<double, double> _betas,
          double _eps,
          double _weight_decay,
          bool _amsgrad,
          bool _bias_correction,
          bool _eps_inside_sqrt,
          double _max_grad_norm,
          bool _use_mt,
          double _amp_scale_adjustment,
          bool _overlap_reductions,
          bool _full_pipeline,
          bool _compute_L2_grad_norm,
          long _dwu_group_size,
          long _dwu_num_blocks,
          long _dwu_num_chunks,
          long _dwu_num_rs_pg,
          long _dwu_num_ar_pg,
          long _dwu_num_ag_pg,
          bool _dwu_exp_enabled,
          long _dwu_exp_num_rs_pg,
          long _dwu_exp_num_ar_pg,
          bool _flat_mt,
          bool _predivide,
          bool _e5m2_allgather,
          bool _do_not_flatten_model);
    ~DistributedFusedAdam();
    void set_last_step(bool last_step);
    void set_global_scale(double global_scale);
    double global_scale();
    bool has_overflow();
    bool peek_overflow();
    float L2_grad_norm();
    void complete_reductions();
    // FIXME: this step() doesn't override the inherited one
    torch::Tensor step(LossClosure closure, bool skip_overflow_check);

    /** FIXME: PyTorch doesn't have C++ scheduler, the solution now is to
     *  use a proxy LR scheduler at Python side, then set/get the LR of
     *  distributed optimizer every iteration.
     */
    float lr(float _lr);

  protected:
#ifndef DIST_OPT_HOOK_TENSOR
    friend class AccGradPostHook;
#endif
    std::pair<long, long> get_flush_block();  // (start, end)
    void pipeline_block_reductions(long block_id);
    void launch_step_kernel(at::Tensor p, at::Tensor p_copy, at::Tensor m,
      at::Tensor v, at::Tensor g);
    void pipeline_block_step(long block_id);
    void pipeline_step();
    void flatten_grad_mt(float scale);
    void do_overlapped_reduction(long param_i, long param_grads_size,
      long param_offset, at::Tensor &param, const at::Tensor &grad);
    bool _strided_check_finite(at::Tensor output_params, int stride,
      int start, int end, bool clear);
    void revert_step();

    std::thread::id worker_tid;
    std::atomic<bool> _isRunning;  // for worker thread join
    std::mutex _mutex;
    std::condition_variable _cv;
    std::unique_ptr<std::thread> _worker;
    BlockingReaderWriterQueue<std::function<void()> > _queue;
    void _worker_thread();

  private:
    DistributedOptimizerOptions options;

    // Distributed optimizer specifics
    int _current_block;
    int _contiguous_idx;
    bool _last_step = false;
    // Must set global scale first
    double _global_scale = std::numeric_limits<double>::quiet_NaN();
    bool _has_overflow = false;

    at::Tensor _overflow_buf = at::zeros({1}, at::TensorOptions().dtype(at::kInt)
      .device(at::kCUDA));
    at::Tensor _L2_grad_norm = at::zeros({1}, at::TensorOptions().dtype(at::kFloat)
      .device(at::kCUDA));
    float _L2_grad_norm_cpu;
    bool _L2_grad_norm_ready = false;

    // Pair of (param_grads_size, param_offset)
    std::vector<std::pair<int64_t, int64_t> > grads_info;
    std::vector<at::Tensor> model_params;
#ifndef DIST_OPT_HOOK_TENSOR
    std::vector<std::shared_ptr<torch::autograd::Node> > grad_accs;
#endif
    std::vector<std::vector<at::Tensor> > grads;
    std::vector<bool> grads_generated;
    // Param index that will invoke gradient reductions
    std::vector<long> low_param_i;

    // Flattened parameters, gradients, states
    at::Tensor flat_grads;
    at::Tensor new_params;
    at::Tensor fp32_p;
    at::Tensor fp32_m;
    at::Tensor fp32_v;
    at::Tensor fp16_p;
    at::Tensor fp16_g;
    std::vector<at::Tensor> individual_flat_grads;

    std::vector<at::Tensor> flat_grads_blocks;
    std::vector<std::vector<at::Tensor> > flat_grads_chunks;
    std::vector<std::vector<std::vector<at::Tensor> > > flat_grads_shards;
    std::vector<at::Tensor> new_params_mega_shards;
    std::vector<std::vector<at::Tensor> > new_params_mega_blocks;
    std::vector<std::vector<std::vector<at::Tensor> > > new_params_mega_chunks;
    std::vector<at::Tensor> fp32_p_blocks;
    std::vector<std::vector<at::Tensor> > fp32_p_chunks;
    std::vector<at::Tensor> fp32_m_blocks;
    std::vector<std::vector<at::Tensor> > fp32_m_chunks;
    std::vector<at::Tensor> fp32_v_blocks;
    std::vector<std::vector<at::Tensor> > fp32_v_chunks;
    std::vector<at::Tensor> fp16_p_blocks;
    std::vector<std::vector<at::Tensor> > fp16_p_chunks;
    std::vector<at::Tensor> fp16_g_blocks;
    std::vector<std::vector<at::Tensor> > fp16_g_chunks;

    std::vector<std::vector<at::Tensor> > packed_flat_to_model_params;

    // CUDA streams for NCCL and optimizer
    std::vector<at::cuda::CUDAStream> rs_st;
    std::vector<at::cuda::CUDAStream> ar_st;
    std::vector<at::cuda::CUDAStream> ag_st;
    at::cuda::CUDAStream l2_grad_norm_st;
    at::cuda::CUDAStream completion_st;

    // Process groups for NCCL
    std::vector<std::shared_ptr<ProcessGroupNCCL> > ar_pg;
    std::vector<std::shared_ptr<ProcessGroupNCCL> > rs_pg;
    std::vector<std::shared_ptr<ProcessGroupNCCL> > ag_pg;

    // No default constructor
    std::unique_ptr<ProcessGroupNCCL> l2_grad_norm_pg;

    // Works for pre communication
    std::vector<std::vector<std::shared_ptr<ProcessGroup::Work> > >
      reductions_works;

    // CUDA events for synchronization
    std::vector<std::vector<at::cuda::CUDAEvent> > reduction_start;
    std::vector<at::cuda::CUDAEvent> reduction_finish;
    at::cuda::CUDAEvent l2_grad_norm_event;
    at::cuda::CUDAEvent completion_event;
};

