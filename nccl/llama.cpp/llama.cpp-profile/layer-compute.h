#include <stdatomic.h>
#include <pthread.h>
#include <omp.h>

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include <time.h>

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int abort;         // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
    struct ggml_threadpool * threadpool;
    int ith;
};

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t layer_cpu_compute(struct ggml_cplan * cplan, struct ggml_tensor * node);
