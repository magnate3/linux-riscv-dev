#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_allocator.h>      // device_allocator
#include <thrust/device_ptr.h>            // device_pointer_cast
#include <thrust/execution_policy.h>      // device
#include <thrust/iterator/zip_iterator.h> // make_zip_iterator
#include <thrust/sort.h>                  // stable_sort
#include <thrust/tuple.h>                 // make_tuple

#define CUDA_CHECK(err)                                                                                      \
    {                                                                                                        \
        if (err != cudaSuccess) {                                                                            \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": "                     \
                      << cudaGetErrorString(err) << std::endl;                                               \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    }

template <typename scale_t, unsigned int TILE_DIM>
__global__ void transpose_kernel(scale_t const *__restrict__ src, scale_t *__restrict__ dst, int rows,
                                 int cols) {
    __shared__ scale_t tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory tile
    for (int j = 0; j < TILE_DIM; j += blockDim.y) {
        // Ensure we don't read out of bounds
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store transposed data from shared memory tile to global memory
    for (int j = 0; j < TILE_DIM; j += blockDim.y) {
        // Ensure we don't read out of bounds
        if (x < rows && (y + j) < cols) {
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename scale_t> struct __align__(4 * sizeof(scale_t)) BBox {
    scale_t x1, y1, x2, y2;

    __host__ __device__ scale_t area() const { return max(scale_t(0), x2 - x1) * max(scale_t(0), y2 - y1); }
    __host__ __device__ static scale_t intersection_area(const BBox &a, const BBox &b) {
        scale_t x1 = max(a.x1, b.x1);
        scale_t y1 = max(a.y1, b.y1);
        scale_t x2 = min(a.x2, b.x2);
        scale_t y2 = min(a.y2, b.y2);
        return max(scale_t(0), x2 - x1) * max(scale_t(0), y2 - y1);
    }
};

template <typename scale_t>
__global__ void split_kernel(scale_t const *__restrict__ src, BBox<scale_t> *__restrict__ bboxes,
                             scale_t *__restrict__ scores, int *__restrict__ class_ids, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }
    // first 4 elements are bbox, but in (cx, cy, w, h) format
    BBox<scale_t> bbox = reinterpret_cast<BBox<scale_t> const *>(src + idx * cols)[0];
    // convert to (x1, y1, x2, y2)
    bboxes[idx] = BBox<scale_t>({bbox.x1 - bbox.x2 * 0.5f, bbox.y1 - bbox.y2 * 0.5f, bbox.x1 + bbox.x2 * 0.5f,
                                 bbox.y1 + bbox.y2 * 0.5f});

    // next elements (cols - 4) are scores for each class
    scale_t max_score = src[idx * cols + 4];
    int max_class_id = 0;
    for (int i = 5; i < cols; ++i) {
        scale_t score = src[idx * cols + i];
        if (score > max_score) {
            max_score = score;
            max_class_id = i - 4;
        }
    }
    scores[idx] = max_score;
    class_ids[idx] = max_class_id;
}

template <typename scale_t> struct Comparator {
    __host__ __device__ inline bool operator()(const thrust::tuple<BBox<scale_t>, scale_t, int> &a,
                                               const thrust::tuple<BBox<scale_t>, scale_t, int> &b) {
        return thrust::get<1>(a) > thrust::get<1>(b);
    }
};

template <typename scale_t>
__device__ bool is_overlapped(BBox<scale_t> const &a, BBox<scale_t> const &b, float iou_threshold) {
    scale_t a_area = a.area();
    scale_t b_area = b.area();
    if (a_area <= 0 || b_area <= 0) {
        return true;
    }
    scale_t inter_area = BBox<scale_t>::intersection_area(a, b);
    scale_t union_area = a_area + b_area - inter_area;
    scale_t iou = inter_area / union_area;
    return iou > iou_threshold;
}

template <typename scale_t>
__global__ void nms_kernel(BBox<scale_t> *__restrict__ bboxes, scale_t *__restrict__ scores,
                           int *__restrict__ class_ids, float iou_threshold, float score_threshold,
                           const int bit_mask_len, int num_boxes, int max_num_boxes,
                           int *accepted_num_boxes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ int mask[];

    // set global mask to accept all boxes (0: accepted, 1: masked/rejected)
    for (int box = idx; box < bit_mask_len; box += stride) {
        mask[box] = 0u;
    }
    __syncthreads();

    int accepted_boxes = 0;
    for (int box = 0; box < num_boxes; ++box) {
        // if current box is masked by an earlier box, skip it.
        if ((mask[box >> 5] >> (box & 31)) & 1) {
            continue;
        }

        for (int b = idx; b < num_boxes; b += stride) {
            if (b <= box)
                continue;
            // if score is below threshold or iou is above threshold, mask it
            if (scores[b] < score_threshold ||
                (class_ids[box] == class_ids[b] &&
                 is_overlapped<scale_t>(bboxes[box], bboxes[b], iou_threshold))) {
                // mask[b >> 5] |= 1u << (b & 31);
                atomicOr(&mask[b >> 5], 1u << (b & 31));
            }
        }
        __syncthreads();

        // shift accepted box to the front of arrays
        if (threadIdx.x == 0) {
            bboxes[accepted_boxes] = bboxes[box];
            scores[accepted_boxes] = scores[box];
            class_ids[accepted_boxes] = class_ids[box];
        }

        accepted_boxes += 1;
        if (accepted_boxes >= max_num_boxes)
            break;
    }

    if (threadIdx.x == 0) {
        *accepted_num_boxes = accepted_boxes;
    }
}

template <typename scale_t>
__global__ void scale_boxes_kernel(BBox<scale_t> *__restrict__ bboxes, int num_selected_boxes,
                                   int const *origin_shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_selected_boxes) {
        return;
    }
    float h = 640, w = 640; // preprocessed image size
    float gain = min(h / origin_shape[0], w / origin_shape[1]);
    float pad_h = round((h - gain * origin_shape[0]) / 2.f - 0.1f);
    float pad_w = round((w - gain * origin_shape[1]) / 2.f - 0.1f);
    BBox<scale_t> box = bboxes[idx];
    box.x1 = min(max((box.x1 - pad_w) / gain, scale_t(0)), scale_t(origin_shape[1]));
    box.y1 = min(max((box.y1 - pad_h) / gain, scale_t(0)), scale_t(origin_shape[0]));
    box.x2 = min(max((box.x2 - pad_w) / gain, scale_t(0)), scale_t(origin_shape[1]));
    box.y2 = min(max((box.y2 - pad_h) / gain, scale_t(0)), scale_t(origin_shape[0]));
    bboxes[idx] = box;
}

// https://forums.developer.nvidia.com/t/is-there-a-similar-temporary-allocation-feature-like-cub-for-thrusts-thrust-sort-by-key/312583
// https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu
template <typename T> class CachingAllocator : public thrust::device_allocator<T> {
  public:
    CachingAllocator() {}

    ~CachingAllocator() {
        if (_ptr)
            thrust::device_allocator<T>::deallocate(_ptr, _size);
        _size = 0;
        _ptr = nullptr;
    }

    thrust::device_ptr<T> allocate(size_t n) {
        if (_ptr && _size >= n)
            return _ptr;
        if (_ptr)
            thrust::device_allocator<T>::deallocate(_ptr, _size);
        _size = n;
        _ptr = thrust::device_allocator<T>::allocate(n);
        return _ptr;
    }

    void deallocate(thrust::device_ptr<T> p, size_t n) {
        // Do not deallocate memory here, we will manage it ourselves
    }

  private:
    size_t _size = 0;
    thrust::device_ptr<T> _ptr = nullptr;
};

template <typename scale_t> class PostProcess {
  public:
    PostProcess(int num_classes, int num_boxes) : num_classes(num_classes), num_boxes(num_boxes) {
        if (num_classes <= 0 || num_boxes <= 0) {
            throw std::runtime_error("num_classes and num_boxes must be positive");
        }
        CUDA_CHECK(cudaMalloc(&src, sizeof(scale_t) * num_boxes * (num_classes + 4)));
        CUDA_CHECK(cudaMalloc(&bboxes, num_boxes * sizeof(BBox<scale_t>)));
        CUDA_CHECK(cudaMalloc(&scores, num_boxes * sizeof(scale_t)));
        CUDA_CHECK(cudaMalloc(&class_ids, num_boxes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&accepted_num_boxes, sizeof(int)));
    }
    ~PostProcess() {
        CUDA_CHECK(cudaFree(src));
        CUDA_CHECK(cudaFree(bboxes));
        CUDA_CHECK(cudaFree(scores));
        CUDA_CHECK(cudaFree(class_ids));
        CUDA_CHECK(cudaFree(accepted_num_boxes));
    }

  public:
    void run(scale_t const *input, int const *origin_shape, float iou_threshold, float score_threshold,
             int max_num_boxes) {
        // Assume that all inputs are valid

        /*
        transpose:
          input: [4 + num_classes, num_boxes]
          output: [num_boxes, 4 + num_classes] => src
        */
        dim3 blockDim(16, 16); // 16 * 16 = 256 threads per block
        dim3 gridDim((num_boxes + blockDim.x - 1) / blockDim.x,
                     (num_classes + 4 + blockDim.y - 1) / blockDim.y);
        transpose_kernel<scale_t, 16><<<gridDim, blockDim>>>(input, src, num_classes + 4, num_boxes);
        CUDA_CHECK(cudaGetLastError());

        /*
        split:
          input:
              src: [num_boxes, 4 + num_classes]
          output:
              bboxes: [num_boxes, 4] (cx, cy, w, h) => (x1, y1, x2, y2)
              scores: [num_boxes]
              class_ids: [num_boxes]
        */
        int threads_per_block = 256;
        int num_blocks = (num_boxes + threads_per_block - 1) / threads_per_block;

        split_kernel<scale_t>
            <<<num_blocks, threads_per_block>>>(src, bboxes, scores, class_ids, num_boxes, num_classes + 4);
        CUDA_CHECK(cudaGetLastError());

        /*
        sort:
          input:
              bboxes: [num_boxes, 4] (x1, y1, x2, y2)
              scores: [num_boxes]
              class_ids: [num_boxes]
          output:
              those inputs are sorted by scores in descending order
        */
        // Create the zip iterators for sorting (bbox, score, class_id)
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(bboxes),
                                                                  thrust::device_pointer_cast(scores),
                                                                  thrust::device_pointer_cast(class_ids)));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(bboxes + num_boxes), thrust::device_pointer_cast(scores + num_boxes),
            thrust::device_pointer_cast(class_ids + num_boxes)));
        // Sort in descending order based on scores
        thrust::stable_sort(thrust::device(alloc), begin, end, Comparator<scale_t>());
        CUDA_CHECK(cudaGetLastError());

        /*
        nms:
          input:
              bboxes: [num_boxes, 4] (x1, y1, x2, y2)
              scores: [num_boxes]
              class_ids: [num_boxes]
          output:
              bboxes: [*, 4] (x1, y1, x2, y2)
              scores: [*]
              class_ids: [*]
        */
        constexpr int kBitsPerMaskElement = 32; // 8 * sizeof(int);
        const int bit_mask_len = (num_boxes + kBitsPerMaskElement - 1) / kBitsPerMaskElement;

        nms_kernel<scale_t><<<1, 1024, bit_mask_len * sizeof(int), 0>>>(
            bboxes, scores, class_ids, iou_threshold, score_threshold, bit_mask_len, num_boxes, max_num_boxes,
            accepted_num_boxes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&num_selected_boxes, accepted_num_boxes, sizeof(int), cudaMemcpyDeviceToHost));

        /*
        scale boxes back to original image size
        */

        threads_per_block = 32;
        num_blocks = (num_selected_boxes + threads_per_block - 1) / threads_per_block;
        scale_boxes_kernel<scale_t>
            <<<num_blocks, threads_per_block>>>(bboxes, num_selected_boxes, origin_shape);
        CUDA_CHECK(cudaGetLastError());
    }

  private:
    int num_classes;
    int num_boxes;
    scale_t *src;
    int *accepted_num_boxes;

    CachingAllocator<char> alloc;

  public:
    BBox<scale_t> *bboxes;
    scale_t *scores;
    int *class_ids;
    int num_selected_boxes;
};
