#include "layer-compute.h"

static void layer_weight_gemv_test(void * data, struct ggml_tensor * node) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;
    const struct ggml_cplan  * cplan  = tp->cplan;
    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    ggml_compute_forward_mul_mat(&params, node);
    assert(!cplan->abort_callback);
    ggml_barrier(state->threadpool);
}

uint64_t layer_cpu_compute(struct ggml_cplan * cplan, struct ggml_tensor * node) {
    int n_threads = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;
    // Reset some of the parameters that need resetting
    // No worker threads should be accessing the parameters below at this stage
    threadpool->cplan            = cplan;
    threadpool->current_chunk    = 0;
    threadpool->abort            = -1;
    threadpool->ec               = GGML_STATUS_SUCCESS;
    uint64_t t_start = get_time_ns();
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            layer_weight_gemv_test(&threadpool->workers[omp_get_thread_num()], node);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        layer_weight_gemv_test(&threadpool->workers[0], node);
    }
    assert(threadpool->ec == GGML_STATUS_SUCCESS);
    return get_time_ns() - t_start;
}
