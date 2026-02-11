#include "layer-compute.cuh"

static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2 + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);

    const float  d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

static void quantize_row_q8_1_cuda(
        const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst) {
    // if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
    switch (ncols_dst) {
        case 1:
            return 1;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
    }
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst) {
    switch (ncols_dst) {
        case 1:
        case 2:
        case 3:
        case 4:
            return 4;
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
    }
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        // sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product
        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF));

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        // sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
        sumf_m += d8[i] * __dp4a(m, u[i], 0);
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

template <int ncols_dst>
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(calc_nwarps(ncols_dst)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nchannels_y, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI2_K;
    constexpr int vdr = VDR_Q2_K_Q8_1_MMVQ;
    constexpr int nwarps = calc_nwarps(ncols_dst);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const int channel_dst = blockIdx.y;
    const int channel_x   = channel_dst / channel_ratio;
    const int channel_y   = channel_dst;
    const int sample_dst  = blockIdx.z;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q2_K_q8_1(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + int(threadIdx.x) < stride_col_dst)) {
            dst[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

extern "C" uint64_t layer_gpu_compute(ggml_tensor * src0_cpu, ggml_tensor * src1_cpu, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, void * context, void * data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) context;
    cudaStream_t stream = cuda_ctx->stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // copy src0 to gpu backend
    CUDA_CHECK(cudaMemcpyAsync((char *)src0->data, src0_cpu->data, ggml_nbytes(src0), cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    // copy src1 to gpu backend
    CUDA_CHECK(cudaMemcpyAsync((char *)src1->data, src1_cpu->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    // static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)
    /* static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) */
    // static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst)
    // static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)
    ///const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft); = false
    ///bad_padding_clear=false; use_mul_mat_vec=false;
    /*bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE; = true*/
    /*bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32; = true*/
    // const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // use_mul_mat_q = use_mul_mat_q && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]); = true
    // any_gpus_with_slow_fp16 = !fast_fp16_hardware_available(cc); = false
    // any_gpus_without_fp16_mma = !fp16_mma_hardware_available(cc); = false
    ///} else if (!split && use_mul_mat_vec_q) ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    /*void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)*/
    uint64_t t_start = get_time_ns();
    assert(src1->type == GGML_TYPE_F32 && dst->type  == GGML_TYPE_F32);
    GGML_TENSOR_BINARY_OP_LOCALS;
    // If src0 is a temporary compute buffer, clear any potential padding.
    ///here we don't have padding, so no need to clear
    // const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);
    ggml_cuda_pool_alloc<char> src1_q8_1(cuda_ctx->pool(), ne13*ne12 * ne11*ne10 * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, src1_q8_1.get(), ne10, s11, s12, s13, ne10, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10 / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;
    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;
    /*static void mul_mat_vec_q_switch_type(
        const void * vx=src0->data, const ggml_type type_x=src0->type, const void * vy=src1_q8_1.get(), const int32_t * ids=nullptr, float * dst=(float *) dst->data,
        const int ncols_x=ne00, const int nrows_x=ne01, const int ncols_dst=ne1,
        const int stride_row_x=s01, const int stride_col_y=s11, const int stride_col_dst=s1,
        const int nchannels_x=ne02, const int nchannels_y=ne12, const int nchannels_dst=ne2,
        const int stride_channel_x=s02, const int stride_channel_y=s12, const int stride_channel_dst=s2,
        const int nsamples_x=ne03, const int nsamples_dst=ne3, const int stride_sample_x=s03, const int stride_sample_y=s13, const int stride_sample_dst=s3,
        cudaStream_t stream=stream) */
    //case GGML_TYPE_Q2_K:
    /*template <ggml_type type> // GGML_TYPE_Q2_K
    static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {*/

    // const int device = ggml_cuda_get_device(); = 0
    // const int warp_size = ggml_cuda_info().devices[device].warp_size; = 32
    // const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc); = 0
    const int channel_ratio = ne2 / ne02;
    const int sample_ratio  = ne3 / ne03;
    ///switch (ncols_dst) { //ncols_dst=1
    constexpr int c_ncols_dst = 1;
    const int64_t nblocks = (ne01 + calc_rows_per_block(c_ncols_dst) - 1) / calc_rows_per_block(c_ncols_dst);
    const dim3 block_nums(nblocks, ne2, ne3);
    const dim3 block_dims(32, calc_nwarps(c_ncols_dst), 1);
    std::pair<dim3, dim3> dims = {block_nums, block_dims};

    mul_mat_vec_q<c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
        (src0->data, src1_q8_1.get(), (float *) dst->data, ne00, ne12, s01, s11, s1,
            channel_ratio, s02, s12, s2,
            sample_ratio, s03, s13, s3);

    cudaStreamSynchronize(stream);
    uint64_t t_ns = get_time_ns() - t_start;
    ggml_backend_tensor_get(dst, data, 0, ggml_nbytes(dst));
    return t_ns;
}
