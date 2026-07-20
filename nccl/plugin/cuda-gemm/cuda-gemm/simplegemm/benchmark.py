import functools
import os
import sys
import numpy as np

import torch
import triton
import triton.language as tl
#import triton.intraprof as proton

#from matmul_persistent_tma_ws_cooperative import matmul_persistent_tma_ws_cooperative

SLOTS = 3*64

torch._dynamo.config.recompile_limit = 1000
torch._inductor.config.max_autotune_gemm_backends = "CUTLASS"
torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
torch._inductor.config.cuda.cutlass_dir = f"{os.environ['HOME']}/local/cutlass"
torch._inductor.config.cuda.cutlass_op_allowlist_regex = "128x128x64_1x1x1.*pingpong_epi_tma"
torch._inductor.config.cuda.cutlass_instantiation_level = "0201"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.", )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.", )


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]



"""
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
        # triton.Config(
        #     {
        #         "BLOCK_SIZE_M": 64,
        #         "BLOCK_SIZE_N": 64,
        #         "BLOCK_SIZE_K": 128,
        #         "GROUP_SIZE_M": 8,
        #         "NUM_CONSUMER_GROUPS": 1,
        #     },
        #     num_stages=3,
        #     num_warps=4,
        #     num_consumer_groups=0, # disable warp specialization
        #     num_buffers_warp_spec=3,
        # ),
    ],
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
"""

@triton.jit
def matmul_persistent_tma_ws_pingpong_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    #profile_mem,
    BLOCK_SIZE_M: tl.constexpr = 128,
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_K: tl.constexpr = 64,  #
    GROUP_SIZE_M: tl.constexpr = 8,  #
    NUM_CONSUMER_GROUPS: tl.constexpr= 1,
):

    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        offs_k0 = 0
        acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a0 = tl._experimental_descriptor_load(
                a_ptr,
                [offs_am, offs_k0],
                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                tl.bfloat16,
            )
            b0 = tl._experimental_descriptor_load(b_ptr, [offs_bn, offs_k0], [BLOCK_SIZE_N, BLOCK_SIZE_K], tl.bfloat16)
            acc0 = tl.dot(a0, b0.T, acc0)
            offs_k0 += BLOCK_SIZE_K

        c0 = acc0.to(tl.bfloat16)
        tl._experimental_descriptor_store(c_ptr, c0, [offs_am, offs_bn])


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

BIN = None


def matmul_persistent_tma_ws_pingpong(a, b, dump_chrome_trace=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    #NUM_SMS=1
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            b.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )
        return (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    global BIN

    def gen_meta(**kwargs):
        return kwargs

    meta = gen_meta(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        NUM_CONSUMER_GROUPS=1,
        num_stages=6,
        num_warps=4,
        num_consumer_groups=1,
        num_buffers_warp_spec=6,
    )
    launch_grid = grid(meta)
    # if dump_chrome_trace:
    #     pconfig = proton.IntraKernelConfig(num_warps=12, proton_slots=SLOTS)
    #     proton_grid = proton.const_grid(launch_grid, autotune_configs=[], func_args={},
    #                                     num_stages=6,
    #                                     num_consumer_groups=1,
    #                                     num_buffers_warp_spec=6,
    #                                     num_warps=4,
    #                                     )
    #     profile_size = proton.intra_kernel_memsize(np.prod(proton_grid), pconfig)
    #     profile_mem = torch.empty(profile_size, device="cuda", dtype=torch.uint32)
    # else:
    #     profile_mem = torch.empty(1, device="cuda", dtype=torch.uint32)
    BIN = matmul_persistent_tma_ws_pingpong_kernel[launch_grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        #profile_mem,
        **meta,
        #proton_slots=SLOTS,
    )
    #if dump_chrome_trace:
    #if True:
    if dump_chrome_trace:
        #print(profile_mem.view(-1, 4))
        proton.dump_chrome_trace(NUM_SMS, pconfig, profile_mem, "chrome_trace.json", BIN)
    return c


def aten_matmul(a, b):
    return a.mm(b)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def cutlass_matmul(a, b):
    return a.mm(b)

torch.ops.load_library("gemm.so")

def custom_gemm(a, b):
    return torch.ops.gemm.gemm(a, b)

def custom_pingpong(a, b):
    return torch.ops.gemm.pingpong(a, b)

def custom_stmatrix_gemm(a, b):
    return torch.ops.gemm.stmatrix_gemm(a, b)

test_impls = [
    aten_matmul,
    cutlass_matmul,
    #custom_gemm,
    custom_pingpong,
    #custom_stmatrix_gemm,
    #matmul_persistent_tma_ws_pingpong,
]

impl_map = {fn.__name__: fn for fn in test_impls}


def test():
    torch.manual_seed(0)
    m = 4 * 11 * 64
    n = 3 * 12 * 256
    #m, n = 2 * 128, 128
    k = 64 * 4
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).T
    torch_output = torch.matmul(a, b)
    rtol = 0
    for fn in test_impls:
        if "cutlass" in fn.__name__:
            continue
        triton_output = fn(a, b)
        torch.cuda.synchronize()
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
            print(f" Torch matches {fn.__name__}")
        else:
            print(f" Torch DOES NOT match {fn.__name__}")
            print("torch output:")
            print(torch_output)
            print("triton output:")
            print(triton_output)


#x_vals = [(8192, 8192, k) for k in range(128, 1280 + 1, 128)]
#x_vals = [(6 * 11 * 128, 3 * 12 * 256, k) for k in range(640, 640 + 1, 128)]
#x_vals = [(4 * 11 * 128, 2 * 12 * 256, k) for k in range(640, 640 + 1, 128)]
#x_vals = [(4 * 11 * 128, 2 * 12 * 256, k) for k in range(128, 2048 + 1, 128)]

#x_vals = [(6 * 11 * 128, 3 * 12 * 256, k) for k in range(128, 2048 + 1, 128)]
#x_vals = [(6 * 11 * 128, 3 * 12 * 256, k) for k in range(640, 640 + 1, 128)]
x_vals = [
    (8192, 8192, 8192),
]
x_vals = [
    (6 * 11 * 128, 6 * 12 * 128, 64 * k)
    for k in range(1, 32)
]

#[
#    (6 * 11 * 128, 6 * 12 * 128, 640),
#    (6 * 11 * 128, 6 * 12 * 128, 1280),
#]
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["K"],  # Argument names to use as an x-axis for the plot
        x_vals=[64 * k for k in range(1, 32)],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[fn.__name__ for fn in test_impls],
        line_names=[
            "Torch (cuBLAS)",
            "Cutlass (no clusters)",
            "Custom CUDA",
        ],
        # styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="pingpong-gemm-performance-bf16",
        args={"M": 6 * 11 * 128, "N": 6 * 12 * 128},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    quantiles = [0.5, 0.2, 0.8]
    fn = impl_map[provider]
    #ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: fn(a, b), quantiles=quantiles)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(a, b), quantiles=quantiles)
    #if provider == "matmul_ws_automatic":
    #    print(getattr(matmul_persistent_tma_ws_cooperative_kernel, "best_config", "not autotune"))
    #    print(BIN.asm["ttgir"])
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)
    #return ms, max_ms, min_ms


def prof(M, N, K, provider="matmul_persistent_tma_ws_pingpong"):
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    #kwargs = {"dump_chrome_trace": True} if provider is "matmul_ws_automatic" else {}
    impl_map[provider](a, b)


def trace():
    M, N, K = 4 * 11 * 128, 4 * 12 * 128, 640
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    for _  in range(3):
        matmul_ws_automatic(a, b)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            matmul_ws_automatic(a, b)

    torch.cuda.synchronize()
    from torch.profiler import profile
    with profile() as p:
        g.replay()
        torch.cuda.synchronize()
    p.export_chrome_trace("prof.json")

#test()
benchmark.run(show_plots=True, print_data=True, save_path=".")
#prof(6 * 11 * 128, 6 * 12 * 128, 1280, provider="cutlass_matmul")
#prof(6 * 11 * 128, 6 * 12 * 128, 1280, provider="custom_pingpong")

print("OK")
