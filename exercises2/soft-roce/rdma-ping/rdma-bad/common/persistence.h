#pragma once

#include "common.h"
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/queue.h>
#include <unistd.h>

enum persistence_output_flags {
    PERSISTENCE_F_FILE = 1U << 0,
    PERSISTENCE_F_STDOUT = 1U << 1,
};

/**
 * Flags defining what should be persisted.
 */
enum persistence_measurement_flags {
    // Store all rounds timestamps. Default option.
    PERSISTENCE_M_ALL_TIMESTAMPS = 1U << 2,
    // Store rounds with minimum and maximum latency
    PERSISTENCE_M_MIN_MAX_LATENCY = 1U << 3,
    // TODO: Store rounds in buckets
    PERSISTENCE_M_BUCKETS = 1U << 4,
};

/**
 * Convert the index of the flag to the value of the flag.
 * This function is useful when the user provides the index of the flag as an argument of the experiment program.
 *
 * @param value the index of the flag
 * @return the value of the flag
 */
inline int pers_measurement_to_flag (const int value)
{
    switch (value)
    {
    case 0:
        return PERSISTENCE_M_ALL_TIMESTAMPS;
    case 1:
        return PERSISTENCE_M_MIN_MAX_LATENCY;
    case 2:
        return PERSISTENCE_M_BUCKETS;
    default:
        return -1;
    }
}

struct min_max_latency_data {
    uint64_t min;
    uint64_t max;
    struct pingpong_payload min_payload;
    struct pingpong_payload max_payload;
};

/* Range in nanoseconds of each bucket */
#define NUM_BUCKETS 20000
#define OFFSET 1000000

struct bucket {
    uint64_t rel_latency[4];
    uint64_t abs_latency;
};

struct bucket_data {
    uint64_t tot_packets;
    uint64_t send_interval;
    struct bucket min_values;
    struct bucket max_values;

    // The layout in memory is as follows:
    // [BUCKET 0 SEND PING] [BUCKET 0 RECV PING] [BUCKET 0 SEND PONG] [BUCKET 0 RECV PONG] [BUCKET 0 LATENCY]
    // The layout is handled manually for convenience.
    union {
        uint64_t *ptr;
        struct bucket *buckets;
    };
    struct pingpong_payload prev_payload;
};

/**
 * Data used by a base file persistence agent.
 */
typedef struct pers_base_data {
    // Output stream to write to
    FILE *file;

    /**
     * Auxiliary data, depending on the flags.
     * - PERSISTENCE_M_ALL_TIMESTAMPS: NULL
     * - PERSISTENCE_M_MIN_MAX_LATENCY: struct min_max_latency_data
     * - PERSISTENCE_M_BUCKETS: struct bucket_data
     */
    void *aux;
} pers_base_data_t;

/**
 * "Persistence agent" to be used to store information about the pingpong measurements.
 */
typedef struct persistence_agent {
    /**
     * Pointer to the data structure used by the persistence agent depending on the flags.
     * This pointer is owned by the persistence agent and should not be freed by the user.
     */
    pers_base_data_t *data;

    /**
     * Flags passed to the initialization function.
     */
    uint32_t flags;

    /**
     * Write data to the persistence agent.
     *
     * @param agent the agent to use
     * @param data the data to write
     * @return 0 on success, -1 on error
     */
    int (*write) (struct persistence_agent *agent, const struct pingpong_payload *data);

    /**
     * Close and cleanup the persistence agent and its data.
     * This function takes ownership of the pointer and frees it. After this function returns,
     * the pointer is no longer valid.
     *
     * @param agent the persistence agent to close
     * @return 0 on success, -1 on error
     */
    int (*close) (struct persistence_agent *agent);
} persistence_agent_t;

/**
 * Initialize persistence module.
 * This function should be called before any other persistence function and only once.
 *
 * @param filename the name of the file to store data
 * @param flags flags to define the type of measurement to store
 * @param aux auxiliary data to be used by the persistence agent
 * @return 0 on success, -1 on error
 */
persistence_agent_t *persistence_init (const char *filename, uint32_t flags, void *aux);