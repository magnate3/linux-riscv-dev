#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <x86intrin.h>
#include <stdatomic.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include "profiler.h"
#include "buffer_pool.h"
#include "output_file.h"

#define __hidden __attribute__((visibility("hidden")))
#define GROUP_POOL_SIZE 64
#define COLL_POOL_SIZE 64
#define P2P_POOL_SIZE 64
#define PROXY_POOL_SIZE 64

#define MAX_CHANNELS 32
#define MAX_STEPS 16
#define MAX_OPS 16 // Up to 64K ranks for PAT

#define NUM_BUCKETS 8

static const char plugin_name[32] = "NCCLSEE Profiler";
static const int defaultEActivationMask = ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp;

static const int64_t buckets[NUM_BUCKETS - 1] = {128, 1024, 8192, 65536, 262144, 1048576, 33554432};

char timestamp_str[32]; // Sufficient for "YYYYMMDD_HHMMSS" + null terminator

/* uint64_t timestamps[1024] = {0}; */
uint32_t event_counter = 0;

enum nccl_colls
{
  nccl_allreduce,
  nccl_broadcast,
  nccl_reduce,
  nccl_reduce_scatter,
  nccl_allgather,
  nccl_alltoall,
  nccl_unknown,  // For unexpected cases
  nccl_num_colls // Keeps track of total primitives
};

static const char *nccl_coll_names[nccl_num_colls] = {
    "AllReduce",
    "Broadcast",
    "ReduceScatter",
    "Reduce",
    "AllGather",
    "AllToAll",
    "Unknown_Collective",
};

enum nccl_p2p
{
  nccl_p2p_send,
  nccl_p2p_recv,
  nccl_p2p_unknown, // For unexpected cases
  nccl_num_p2p      // Keeps track of total primitives
};

static const char *nccl_p2p_names[nccl_num_p2p] = {
    "Send",
    "Recv",
    "Uknown_P2P"};

/* static const int groupPoolSize = 128; */
/* static const int collPoolSize = 128; */

static struct
{
  uint64_t count;
  uint64_t bytes;
  //_Atomic double time;
  _Atomic double time;
  // We may add more things here
} stats[nccl_num_colls][NUM_BUCKETS] = {0};

static struct
{
  uint64_t count;
  uint64_t typecount;
  //_Atomic double time;
  _Atomic double time;
} stats_p2p[nccl_num_p2p][NUM_BUCKETS] = {0};

static struct
{
  uint64_t count;
  //_Atomic double time;
  _Atomic double time;
} stats_group = {0};

struct context;

struct group
{
  uint8_t type;
  struct context *ctx;
  int refCount;
  double startTs;
  double stopTs;
};

// task level event base structure
struct taskEventBase
{
  uint8_t type; // event type: collective/p2p
  // int rank;                         // rank of the operation in NCCL communicator
  // uint64_t commHash;                // communicator identifier
  const char *func;     // ncclFunc*
  int refCount;         // number of references for this operation
  struct group *parent; // parent event group
  double startTs;
  double stopTs;
};

struct proxyOp
{
  uint8_t type; // ncclProfileProxyOp
  pid_t pid;
  struct taskEventBase *parent; // parent event p2p/collective
};

struct collective
{
  struct taskEventBase base;
  enum nccl_colls name; // Index in the collective name array
  int bucket_index;    // We probably dont need this
  // struct proxyOp send[MAX_CHANNELS][MAX_OPS];// array of send proxy operation events
  // struct proxyOp recv[MAX_CHANNELS][MAX_OPS];// array of recv proxy operation events
  // int nProxyOps[MAX_CHANNELS];
};

struct p2p
{
  struct taskEventBase base;
  enum nccl_p2p name;
  int bucket_index;
  // struct proxyOp op[MAX_CHANNELS];
};

struct context
{
  int groupIndex;
  struct group groupPool[GROUP_POOL_SIZE];
  int collIndex;
  struct collective collPool[COLL_POOL_SIZE];
  int p2pIndex;
  struct p2p p2pPool[P2P_POOL_SIZE];
  int proxyIndex;
  struct proxyOp proxyPool[PROXY_POOL_SIZE];
};

static uint8_t cupti_buffer[TOTAL_POOL_SIZE]; // 32KB buffer for CUPTI activity records.

static int initialized = 0; // initialization counter for profiler
static int output = 0;
static FILE *debug_file = NULL;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pid_t pid;
static double startTime;

static double freq = -1;

// Simple correlation tracker
typedef struct
{
  enum nccl_colls opType;
  int bucketIndex;
  bool valid;
} CorrelationInfo;

#define MAX_CORRELATIONS 16384 // Adjust based on expected concurrent operations

// Simple correlation tracker
static CorrelationInfo correlationTracker[MAX_CORRELATIONS] = {0};


// Store correlation info
void storeCorrelation(uint64_t correlationId, enum nccl_colls opType, int bucketIndex)
{
  // Simple modulo-based index to handle wrap-around
  int index = correlationId % MAX_CORRELATIONS;
  correlationTracker[index].opType = opType;
  correlationTracker[index].bucketIndex = bucketIndex;
  correlationTracker[index].valid = true;
#ifdef DEBUG
  fprintf(debug_file, "Correlation %u: %s, %d\n", correlationId, nccl_coll_names[opType], bucketIndex);
  fflush(debug_file);
#endif
}

// Retrieve correlation info
bool getCorrelation(uint64_t correlationId, enum nccl_colls *opType, int *bucketIndex)
{
  int index = correlationId % MAX_CORRELATIONS;
  if (correlationTracker[index].valid)
  {
    *opType = correlationTracker[index].opType;
    *bucketIndex = correlationTracker[index].bucketIndex;
    return true;
  }
  return false;
}

// Generate a unique correlation ID
uint64_t generateCorrelationId()
{
  static uint64_t correlationId = 0;
  return correlationId++;
}

// Subscribe to CUDA runtime API callbacks
CUpti_SubscriberHandle subscriber;

__hidden void calibrate()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = -tv.tv_sec * 1e6 - tv.tv_usec;
  uint64_t total = 0ULL;
  // Dummy loop to let some time pass
  for (int i = 0; i < 10000; i++)
    total += __rdtsc();

  gettimeofday(&tv, NULL);

  timeCycles = __rdtsc() - timeCycles;  // Compute elapsed cycles
  time += tv.tv_sec * 1e6 + tv.tv_usec; // Compute elapsed real-world time
  freq = timeCycles / time;
}


__hidden void calibrate_ns() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = - tv.tv_sec*1e9 - tv.tv_usec*1000; // Convert to nanoseconds
  uint64_t total = 0ULL;
  for (int i = 0; i < 10000; i++) total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec*1e9 + tv.tv_usec*1000; // Convert to nanoseconds
  freq = timeCycles / time; // Now freq is cycles per nanosecond
}

// returns current timestamp in useconds
__hidden double gettime_ns(void)
{
  // return __rdtsc() / freq;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((ts.tv_sec * 1e9) + ts.tv_nsec);
}

double gettime()
{
  return (double)__rdtsc() / freq;
}

static int getNCCLTypeSize(const char *datatype)
{
  if (datatype == NULL)
  {
    // Error: datatype is NULL print error message
    fprintf(stderr, "ncclsee Error: NULL datatype passed to getNCCLTypeSize\n");
    // Return -1 to indicate error, the caller should check for this
    return -1;
  }

  // 1-byte types
  if (strcmp(datatype, "ncclInt8") == 0 ||
      strcmp(datatype, "ncclUint8") == 0 ||
      strcmp(datatype, "ncclFloat8e4m3") == 0 ||
      strcmp(datatype, "ncclFloat8e5m2") == 0)
  {
    return 1;
  }

  // 2-byte types
  if (strcmp(datatype, "ncclFloat16") == 0 ||
      strcmp(datatype, "ncclBfloat16") == 0)
  {
    return 2;
  }

  // 4-byte types
  if (strcmp(datatype, "ncclInt32") == 0 ||
      strcmp(datatype, "ncclUint32") == 0 ||
      strcmp(datatype, "ncclFloat32") == 0)
  {
    return 4;
  }

  // 8-byte types
  if (strcmp(datatype, "ncclInt64") == 0 ||
      strcmp(datatype, "ncclUint64") == 0 ||
      strcmp(datatype, "ncclFloat64") == 0)
  {
    return 8;
  }

  // Unknown type
  fprintf(stderr, "ncclsee Error: Unknown datatype passed to getNCCLTypeSize\n");
  return 0; // Default to 0 bytes for unknown types
}

int get_nccl_coll_name(const char *name)
{

  if (strcmp(name, "AllReduce") == 0)
  {
    return nccl_allreduce;
  }
  else if (strcmp(name, "Broadcast") == 0)
  {
    return nccl_broadcast;
  }
  else if (strcmp(name, "Reduce") == 0)
  {
    return nccl_reduce;
  }
  else if (strcmp(name, "ReduceScatter") == 0)
  {
    return nccl_reduce_scatter;
  }
  else if (strcmp(name, "AllGather") == 0)
  {
    return nccl_allgather;
  }
  else if (strcmp(name, "AllToAll") == 0)
  {
    return nccl_alltoall;
  }
  // Keeping track of this for now for debugging purposes
  else
  {
    return nccl_unknown;
  }
}

// Find the appropriate bucket for a given byte size
int choose_bucket(int64_t bytes)
{
  for (int index = 0; index < NUM_BUCKETS - 1; ++index)
  {
    if (buckets[index] > bytes)
    {
      return index;
    }
  }
  return NUM_BUCKETS - 1;
}

void atomic_add_double(_Atomic double *target, double increment)
{
  double current;
  double desired;
  do
  {
    current = atomic_load(target);
    desired = current + increment;
  } while (!atomic_compare_exchange_weak(target, &current, desired));
}

#define CUPTI_CALL(call)                                                   \
  do                                                                       \
  {                                                                        \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS)                                          \
    {                                                                      \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

// CUPTI callback: Allocate a buffer for activity records.
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  *size = POOL_BUFFER_SIZE;
  size_t actual_size = 0;
  *buffer = get_buffer_from_pool(&actual_size);
  //*buffer = cupti_buffer;
  if (*buffer == NULL ||  actual_size != POOL_BUFFER_SIZE)
  {
    fprintf(stderr, "Error: unable to allocate CUPTI buffer\n");
    exit(-1);
  }
  *maxNumRecords = 0;
}


void CUPTIAPI bufferCompleted(CUcontext context, uint32_t streamId,
                              uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  uint64_t previous_external_id = 0;
  uint32_t previous_correlation_id = 0;

  do
  {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status != CUPTI_SUCCESS)
      break;

    if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)
    {
      CUpti_ActivityExternalCorrelation *correlation =
          (CUpti_ActivityExternalCorrelation *)record;
      previous_external_id = correlation->externalId;
      previous_correlation_id = correlation->correlationId;
    }
    else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
    {
      CUpti_ActivityKernel8 *kernel = (CUpti_ActivityKernel8 *)record;
      if (previous_correlation_id == kernel->correlationId){
        // We only care about NCCL kernels
        if (strstr(kernel->name, "nccl") == NULL)
          continue;
        enum nccl_colls opType;
        int bucketIndex;
        getCorrelation(previous_external_id, &opType, &bucketIndex);
        double kernelTime = (kernel->completed - kernel->start) / 1e3; //Convert to microseconds
        atomic_add_double(&stats[opType][bucketIndex].time, kernelTime);
        printf("KERNEL: name=%s\n", kernel->name);
      }
      else{
        /* printf("KERNEL: name=%s, correlationId=%u, previous_external_id=%lu, previous_correlation_id=%u \n", */
        /*      kernel->name, kernel->correlationId,previous_external_id,previous_correlation_id); */
      }

    }
    else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME)
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;

      printf("RUNTIME API: cbid=%u, correlationId=%u\n",
             api->cbid, api->correlationId);
    }
  } while (1);
  return_buffer_to_pool(buffer);
  //free(buffer);
}

// Dont forget to register the function in Profiler_Init
void CUPTIAPI cupti_callback_handler(void *userdata, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const void *cbdata)
{
  const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;

  // Only process CUDA Runtime API calls
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API)
  {
    return;
  }

  // Check if this is a cudaDeviceSynchronize exit
  if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 &&
      cbInfo->callbackSite == CUPTI_API_EXIT)
  {

    // Flush CUPTI activity data after synchronization
    CUptiResult status = cuptiActivityFlushAll(0);
    if (status != CUPTI_SUCCESS)
    {
      const char *errstr;
      cuptiGetResultString(status, &errstr);
      fprintf(stderr, "Error flushing CUPTI activities after cudaDeviceSynchronize: %s\n", errstr);
    }

    // Process the flushed data here or in the bufferCompleted callback
    // A good time to update timing statistics
#ifdef DEBUG
    fprintf(debug_file, "Flushed CUPTI activities after cudaDeviceSynchronize\n");
    fflush(debug_file);
#endif
  }
}

__hidden ncclResult_t Profiler_Init(void **context, int *eActivationMask)
{


  struct tm *local_time;
  time_t now;


  // 1. Get current time and format it
  now = time(NULL);
  if (now == ((time_t)-1)) {
    perror("Failed to get current time");
    return ncclInternalError;
  }

  local_time = localtime(&now);
  if (local_time == NULL) {
    perror("Failed to convert time to local time structure");
    return ncclInternalError;
  }

  // We use this timestamp to create the output folder
  if (strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S",
               local_time) == 0) {
    fprintf(stderr, "ncclsee Error: Failed to format timestamp (strftime returned 0)\n");
    return ncclInternalError;
  }


  //cudaDeviceSynchronize();

  pthread_mutex_lock(&lock);
  if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0)
  {
    // first thread initializes event mask, environment and detach pool
    const char *str;
    str = getenv("NCCL_PROFILE_EVENT_MASK");
    __atomic_store_n(eActivationMask, str ? atoi(str) : defaultEActivationMask, __ATOMIC_RELAXED);
    pid = getpid();
    calibrate();
    if ( freq == -1 )
    {
      fprintf(stderr, "Profiler_Init: Error calibrating CPU frequency\n");
      fflush(stderr);
      pthread_mutex_unlock(&lock);
      return ncclInternalError;
    }
    init_buffer_pool(cupti_buffer);
    startTime = gettime();
  }
  pthread_mutex_unlock(&lock);


  CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cupti_callback_handler, NULL));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  // Initialize CUPTI to record kernel events
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  uint64_t correlationId = generateCorrelationId();
  CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
      correlationId));

  /* fprintf(stderr, "Profiler_Init: %s\n",plugin_name); */
  /* fprintf(stderr, "Profiler_Init: eActivationMask = %d\n", *eActivationMask); */
#ifdef DEBUG
  char debug_file_name[64];
  snprintf(debug_file_name, 64, "./ncclsee_debug_%d.log", pid);
  debug_file = fopen(debug_file_name, "a+");
#endif
  // Allocate memory for the context
  struct context *ctx = (struct context *)calloc(1, sizeof(struct context));
  if (ctx == NULL)
  {
    fprintf(stderr, "Profiler_Init: Failed to allocate memory for context\n");
    return ncclInternalError; // Return an appropriate NCCL error code
  }
  ctx->groupIndex = 0;
  ctx->collIndex = 0;
  ctx->p2pIndex = 0;
  ctx->proxyIndex = 0;
  // Assign the context to the output parameter
  *context = ctx;
  /* CUPTI_CALL(cuptiActivityEnableLatencyTimestamps(1)); */

  return ncclSuccess;
}

__hidden ncclResult_t Profiler_Finalize(void *context)
{

  CUPTI_CALL(cuptiActivityFlushAll(0));
  CUPTI_CALL(cuptiUnsubscribe(subscriber));
  CUPTI_CALL(cuptiFinalize());



pthread_mutex_lock(&lock);
  if (__atomic_fetch_add(&output, 1, __ATOMIC_RELAXED) == 0){
    FILE *fp = create_profile_file(timestamp_str);
    if (fp == NULL)
    {
      fprintf(stderr, "ncclsee Error: creating profile file\n");
      pthread_mutex_unlock(&lock);
      return ncclInternalError;
    }
/* fprintf(stderr, "\n=========================== NCCL PROFILING SUMMARY ===========================================\n"); */
/* fprintf(stderr, "%-18s %-18s %-18s %-20s %-15s\n", */
/*         "Collective Type", "Bucket (B)", "Calls", "Bytes Transferred", "Total Time (us)"); */
/* fprintf(stderr, "------------------------------------------------------------------------------------------------\n"); */

  // Example: Write the header
  fprintf(fp, "CollectiveType,BucketLabel,BucketMin,BucketMax,"
                      "Calls,BytesTransferred,TotalTime_us\n");

for (int i = 0; i < nccl_num_colls; i++) {
  /* uint64_t total_count = 0; */
  /* uint64_t total_bytes = 0; */
  /* double total_time = 0.0; */

  for (int j = 0; j < NUM_BUCKETS; j++) {


    if (stats[i][j].count == 0)
      continue;

    // Determine bucket range string
    char bucket_range[20];
    int64_t upper_bound,lower_bound;
    if (j == 0) {
      snprintf(bucket_range, sizeof(bucket_range), "0-%ld", buckets[0]);
      upper_bound = buckets[0];
      lower_bound = 0;

    } else if (j == NUM_BUCKETS - 1) {
      snprintf(bucket_range, sizeof(bucket_range), ">%ld", buckets[NUM_BUCKETS - 2]);
      upper_bound = INT32_MAX;
      lower_bound = buckets[NUM_BUCKETS - 2];

    } else {
      snprintf(bucket_range, sizeof(bucket_range), "%ld-%ld", buckets[j-1], buckets[j]);
      upper_bound = buckets[j];
      lower_bound = buckets[j-1];
    }
    fprintf(fp, "%s,%s,%ld,%ld,%" PRIu64 ",%" PRIu64 ",%.6f\n",
            nccl_coll_names[i], bucket_range, (long)lower_bound, (long)upper_bound,
            (uint64_t)stats[i][j].count, (uint64_t)stats[i][j].bytes,
            stats[i][j].time);

    /* fprintf(stderr, "%-18s %-18s %-18" PRIu64 " %-20" PRIu64 " %-15.6f\n", */
    /*         nccl_coll_names[i], bucket_range, stats[i][j].count, */
    /*         stats[i][j].bytes, stats[i][j].time / 1e3); */


    /* total_count += stats[i][j].count; */
    /* total_bytes += stats[i][j].bytes; */
    /* total_time += stats[i][j].time; */
  }

  // Print totals for this collective type if any calls were made
  /* if (total_count > 0) { */
  /*   fprintf(stderr, "%-18s %-18s %-18" PRIu64 " %-20" PRIu64 " %-15.6f\n", */
  /*           nccl_coll_names[i], "TOTAL", total_count, total_bytes, total_time / 1e3); */
  /*   fprintf(stderr, "------------------------------------------------------------------------------------------------\n"); */
  /* } */
}

/* fprintf(stderr, "%-18s %-18s %-18" PRIu64 " %-20" PRIu64 " %-15.6f\n", */
/*         "Group", "N/A", stats_group.count, (uint64_t)0, stats_group.time / 1e3); */
/* fprintf(stderr, "==================================================================================================\n\n"); */
fprintf(fp, "Group,N/A,0,0,%" PRIu64 ",%" PRIu64 ",%.6f\n",
        (uint64_t)stats_group.count, (uint64_t)0, stats_group.time);
if (fclose(fp) != 0) {
    perror("ncclsee Warning: Failed to close profile file properly");
} else {
    fprintf(stderr, "ncclsee: Successfully wrote and closed profile file for PID %d.\n",
            getpid());
  }
  }
  pthread_mutex_unlock(&lock);


  struct context *ctx = (struct context *)context;
  if (ctx != NULL)
    free(ctx);
  if (debug_file)
    fclose(debug_file);
  destroy_buffer_pool();
  return ncclSuccess;
}

ncclResult_t Profiler_Event_Start(void *context, void **eHandle, ncclProfilerEventDescr_t *eDescr)
{
  *eHandle = NULL;
  struct context *ctx = (struct context *)context;
#ifdef DEBUG
  fprintf(debug_file, "Profiler_Event_Start: %d\n", eDescr->type);
  fflush(debug_file);
#endif

  if (eDescr->type == ncclProfileGroup)
  {
    /* struct group* event = (struct group*)malloc(sizeof(struct group)); */
    /* if (event == NULL) { */
    /*     return ncclInternalError; */
    /* } */
    // Get the next index in the group pool (circular buffer behavior)
    int index = __atomic_fetch_add(&ctx->groupIndex, 1, __ATOMIC_RELAXED) % GROUP_POOL_SIZE;
    struct group *event = &ctx->groupPool[index];
    event->startTs = gettime() - startTime;
    event->ctx = ctx;
    event->type = ncclProfileGroup;
    //__atomic_fetch_add(&stats_group.count, 1, __ATOMIC_RELAXED);
    stats_group.count++;
    //event->startTs = gettime();
    *eHandle = event;
  }
  else if (eDescr->type == ncclProfileColl)
  {

    struct group *parent = (struct group *)eDescr->parentObj;
    if (parent == NULL)
      return ncclSuccess;

    int index = __atomic_fetch_add(&ctx->collIndex, 1, __ATOMIC_RELAXED) % COLL_POOL_SIZE;
    struct collective *event = &ctx->collPool[index];

    event->base.startTs = gettime() - startTime;
    event->base.type = ncclProfileColl;
    size_t trafficBytes = eDescr->coll.trafficBytes;
    event->base.parent = parent;

    const char *name = eDescr->coll.func;

    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    int type_size = getNCCLTypeSize(eDescr->coll.datatype);
    if (type_size == -1 || type_size == 0)
    {
      fprintf(stderr, "ncclsee Profiler_Event_Start: Error getting type size\n");
      // We need to clean up here in the future
      return ncclInternalError;
    }
   
    size_t count = eDescr->coll.count;
    event->name = get_nccl_coll_name(name);
    size_t bufferSize = count * type_size;
    uint64_t correlationId;
    int bucket_index = choose_bucket(bufferSize);

    CUptiResult popStatus = cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
      &correlationId);
  
  if (popStatus == CUPTI_ERROR_QUEUE_EMPTY) {
      // Handle empty queue case - either skip correlation or generate new ID
      correlationId = generateCorrelationId();
  } else if (popStatus != CUPTI_SUCCESS) {
      const char *errstr;
      cuptiGetResultString(popStatus, &errstr);
      fprintf(stderr, "Error popping correlation ID: %s\n", errstr);
      return ncclInternalError;
  }

      // CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
      //     CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
      //     &correlationId));
      storeCorrelation(correlationId, event->name, bucket_index);
      event_counter++;
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
                                                        correlationId));
      generateCorrelationId();

#ifdef DEBUG
    fprintf(debug_file, "Datatype %s\n", eDescr->coll.datatype);
    fflush(debug_file);
#endif
    // It is better to update those now so we dont carry them around

    /* __atomic_fetch_add(&stats[event->name][bucket_index].count, 1, __ATOMIC_RELAXED); */
    /* __atomic_fetch_add(&stats[event->name][bucket_index].bytes, trafficBytes, __ATOMIC_RELAXED); */
    stats[event->name][bucket_index].count++;
    stats[event->name][bucket_index].bytes += trafficBytes;
    event->bucket_index = bucket_index;
    //event->base.startTs = gettime();
    *eHandle = event;
  }
  else if (eDescr->type == ncclProfileP2p)
  {
    struct group *parent = (struct group *)eDescr->parentObj;
    if (parent == NULL)
      return ncclSuccess;

    int index = __atomic_fetch_add(&ctx->p2pIndex, 1, __ATOMIC_RELAXED) % P2P_POOL_SIZE;
    struct p2p *event = &ctx->p2pPool[index];
    event->base.startTs = gettime() - startTime;
    event->base.type = ncclProfileP2p;
    event->base.parent = parent;
    const char *name = eDescr->coll.func;

    if (strcmp(eDescr->p2p.func, "Send") == 0)
    {
      event->name = nccl_p2p_send;
    }
    else if (strcmp(eDescr->p2p.func, "Recv") == 0)
    {
      event->name = nccl_p2p_recv;
    }
    else
    {
      event->name = nccl_p2p_unknown;
    }


    int type_size = getNCCLTypeSize(eDescr->coll.datatype);
    size_t count = eDescr->coll.count;
    event->name = get_nccl_coll_name(name);
    size_t bufferSize = count * type_size;
    uint64_t correlationId;
    int bucket_index = choose_bucket(bufferSize);

      CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
          &correlationId));
      storeCorrelation(correlationId, event->name, bucket_index);
      event_counter++;
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
                                                        correlationId));
      generateCorrelationId();
    

    /* __atomic_fetch_add(&stats_p2p[event->name][bucket_index].count, 1, __ATOMIC_RELAXED); */
    /* __atomic_fetch_add(&stats_p2p[event->name][bucket_index].typecount, eDescr->p2p.count, __ATOMIC_RELAXED); */
    stats_p2p[event->name][bucket_index].count++;
    stats_p2p[event->name][bucket_index].typecount += eDescr->p2p.count;
    //event->base.startTs = gettime();
    *eHandle = event;
  }
  else if (eDescr->type == ncclProfileProxyOp)
  {

    // fprintf(debug_file, "ProxyOp\n");
    struct taskEventBase *eventBase = (struct taskEventBase *)eDescr->parentObj;
    if (eventBase == NULL)
      return ncclSuccess;
    // fprintf(debug_file, "ProxyOp parent not NULL\n");
    if (eDescr->proxyOp.pid != pid)
    {
      int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
      struct proxyOp *event = &ctx->proxyPool[index];
      event->type = ncclProfileProxyOp;
      event->pid = eDescr->proxyOp.pid;
      event->parent = NULL;
      *eHandle = event;
      return ncclSuccess;
    }

    if (eventBase->type == ncclProfileColl)
    {
      // Cannot be NULL
      struct collective *parent = (struct collective *)eDescr->parentObj;
      int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
      struct proxyOp *event = &ctx->proxyPool[index];
      event->type = ncclProfileProxyOp;
      event->pid = eDescr->proxyOp.pid;
      event->parent = eventBase;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      *eHandle = event;
    }
    else
    {
      struct p2p *parent = (struct p2p *)eDescr->parentObj;
      int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
      struct proxyOp *event = &ctx->proxyPool[index];
      event->type = ncclProfileProxyOp;
      event->pid = eDescr->proxyOp.pid;
      event->parent = eventBase;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      *eHandle = event;
    }
  }

  return ncclSuccess;
}

static void updateEvent(void *handle)
{
  if (handle == NULL)
    return;

  uint8_t type = *(uint8_t *)handle;
  if (type == ncclProfileGroup)
  {
    struct group *event = (struct group *)handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0)
    {
      event->stopTs = gettime() - startTime;
      //event->stopTs = gettime();
      double duration = event->stopTs - event->startTs;
      atomic_add_double(&stats_group.time, duration);
      /* stats_group.time += duration; */
    }
  }
  else if (type == ncclProfileColl)
  {
    struct collective *event = (struct collective *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0)
    {
      event->base.stopTs = gettime() - startTime;
      //event->base.stopTs = gettime();
      double duration = event->base.stopTs - event->base.startTs;
      // Update the time in stats
      // fprintf(debug_file, "Collective %s took %lf us\n", nccl_coll_names[event->name], duration);
      atomic_add_double(&stats[event->name][event->bucket_index].time, duration);
      /* stats[event->name][event->bucket_index].time += duration; */
      updateEvent(event->base.parent);

      return;
    }
  }
  else if (type == ncclProfileP2p)
  {
    struct p2p *event = (struct p2p *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0)
    {
      event->base.stopTs = gettime() - startTime;
      //event->base.stopTs = gettime();
      double duration = event->base.stopTs - event->base.startTs;
      // Update the time in stats
      // fprintf(debug_file, "P2P %s took %lf us\n", nccl_p2p_names[event->name], duration);
      atomic_add_double(&stats_p2p[event->name][event->bucket_index].time, duration);
      /* stats_p2p[event->name][event->bucket_index].time += duration; */
      updateEvent(event->base.parent);
      return;
    }
  }
  else if (type == ncclProfileProxyOp)
  {
    struct proxyOp *event = (struct proxyOp *)handle;
    // We are not measuring proxy ops time yet
    updateEvent(event->parent);
  }
}

__hidden ncclResult_t Profiler_Event_Stop(void *eHandle)
{
  if (eHandle == NULL)
    return ncclSuccess;

  uint8_t type = *(uint8_t *)eHandle;
#ifdef DEBUG
  fprintf(debug_file, "Profiler_Event_Stop: %d\n", type);
  fflush(debug_file);
#endif

  if (type == ncclProfileGroup)
  {
    struct group *event = (struct group *)eHandle;
    event->stopTs = gettime() - startTime;
    //event->stopTs = gettime();
    // Update the time in stats atomically
    atomic_add_double(&stats_group.time, event->stopTs - event->startTs);
    //stats_group.time += event->stopTs - event->startTs;
#ifdef DEBUG
    fprintf(debug_file, "Group took %lf us, Accumulated %lf\n", event->stopTs - event->startTs, stats_group.time);
    fflush(debug_file);
#endif
    return ncclSuccess;
  }
  else if (type == ncclProfileColl)
  {
    struct collective *event = (struct collective *)eHandle;
    event->base.stopTs = gettime() - startTime;
    //event->base.stopTs = gettime();
    // Update the time in collective stats atomically
    atomic_add_double(&stats[event->name][event->bucket_index].time, event->base.stopTs - event->base.startTs);
    //stats[event->name][event->bucket_index].time += event->base.stopTs - event->base.startTs;

    // Update the time in case proxy ops are used
    // event->base.startTs = event->base.stopTs;
    /* timestamps[event_counter] = get_timestamp_ns(); */
    /* event_counter++; */
    return ncclSuccess;
  }
  else if (type == ncclProfileP2p)
  {
    struct p2p *event = (struct p2p *)eHandle;
    event->base.stopTs = gettime() - startTime;
    //event->base.stopTs = gettime();
    atomic_add_double(&stats_p2p[event->name][event->bucket_index].time, event->base.stopTs - event->base.startTs);
    /* stats_p2p[event->name][event->bucket_index].time += event->base.stopTs - event->base.startTs; */
    // Update the time in case proxy ops are used
    // event->base.startTs = event->base.stopTs;
    return ncclSuccess;
  }

  updateEvent(eHandle);
  return ncclSuccess;
}

__hidden ncclResult_t Profiler_Event_Record(void *eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t *eStateArgs)
{
  return ncclSuccess;
}

ncclProfiler_t ncclProfiler_v2 = {
    .name = plugin_name,
    .init = Profiler_Init,
    .startEvent = Profiler_Event_Start,
    .stopEvent = Profiler_Event_Stop,
    .recordEventState = Profiler_Event_Record,
    .finalize = Profiler_Finalize};
