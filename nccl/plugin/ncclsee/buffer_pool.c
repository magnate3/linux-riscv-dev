#include "buffer_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// --------------------------------------------------------------------------
// Buffer Pool State (Static/Global within this file)
// These variables hold the internal state of the buffer pool.
// --------------------------------------------------------------------------

// Pointer to the base of the contiguous memory block managed by the pool.
// This block is provided by the caller of init_buffer_pool.
// It is NOT allocated or freed by buffer_pool.c
static uint8_t* base_memory_block = NULL;

// Status array: buffer_status[i] is 1 if buffer slice i is free, 0 if in use.
static int buffer_status[POOL_BUFFER_COUNT];

// Index tracking the next buffer slice to consider in a cyclic (round-robin) manner.
static int next_buffer_idx = 0;

// Mutex to protect access to the shared pool state (buffer_status, next_buffer_idx, base_memory_block).
// Essential for thread safety if get/return are called from multiple threads.
static pthread_mutex_t pool_mutex;

// Flag to indicate if the pool has been successfully initialized.
static int pool_initialized = 0;

// --------------------------------------------------------------------------
// Buffer Pool Management Functions Implementation
// --------------------------------------------------------------------------

// Initialize the buffer pool: Set up pointers within the provided memory block.
void init_buffer_pool(uint8_t* memory_block) {
    // Using a flag and mutex for a more robust check against double initialization
    pthread_mutex_lock(&pool_mutex); // Acquire mutex before checking initialized state
    if (pool_initialized) {
        pthread_mutex_unlock(&pool_mutex);
        fprintf(stderr, "Warning: Buffer pool already initialized.\n");
        return;
    }
    pthread_mutex_unlock(&pool_mutex); // Release after check

    if (memory_block == NULL) {
        fprintf(stderr, "Error: init_buffer_pool called with NULL memory_block.\n");
        exit(EXIT_FAILURE); // Cannot initialize without valid memory
    }

    /* fprintf(stderr, "Initializing buffer pool to manage memory block %p with %d buffers of size %d bytes each.\n", */
    /*         memory_block, POOL_BUFFER_COUNT, POOL_BUFFER_SIZE); */

    // Initialize the mutex BEFORE acquiring it for the first time
    if (pthread_mutex_init(&pool_mutex, NULL) != 0) {
        fprintf(stderr, "Error: Failed to initialize buffer pool mutex.\n");
        exit(EXIT_FAILURE); // Cannot proceed without thread safety
    }

    // Now acquire the mutex for initializing shared state
    pthread_mutex_lock(&pool_mutex);

    // Store the base of the memory block
    base_memory_block = memory_block;

    // Initialize buffer statuses to free
    for (int i = 0; i < POOL_BUFFER_COUNT; ++i) {
        buffer_status[i] = 1; // Mark as free
        // Optional: Initialize buffer content for debugging purposes
        memset(base_memory_block + (size_t)i * POOL_BUFFER_SIZE, 0, POOL_BUFFER_SIZE);
    }

    // Initialize the cyclic index
    next_buffer_idx = 0;
    pool_initialized = 1; // Mark as initialized

    // Release mutex
    pthread_mutex_unlock(&pool_mutex);

    /* fprintf(stderr, "Buffer pool initialized successfully.\n"); */
}

// Destroy the buffer pool. (Just cleans up internal resources like the mutex).
// It DOES NOT free the memory_block provided during init.
void destroy_buffer_pool(void) {
     // Acquire mutex before checking and modifying initialized state
    pthread_mutex_lock(&pool_mutex);
    if (!pool_initialized) {
        pthread_mutex_unlock(&pool_mutex);
        fprintf(stderr, "Warning: Buffer pool not initialized or already destroyed.\n");
        return;
    }

    /* fprintf(stderr, "Destroying buffer pool...\n"); */

    // Mark as uninitialized while holding the mutex
    pool_initialized = 0;
    base_memory_block = NULL; // Clear the pointer to managed memory

    // Release mutex before destroying it
    pthread_mutex_unlock(&pool_mutex);

    // Destroy the mutex
    pthread_mutex_destroy(&pool_mutex);

    /* fprintf(stderr, "Buffer pool destroyed.\n"); */
}

// Get a buffer slice from the pool. Returns NULL if no buffer is available.
uint8_t* get_buffer_from_pool(size_t* actual_size) {
    uint8_t* buffer = NULL;
    int starting_idx;

    if (!pool_initialized) {
         fprintf(stderr, "Error: get_buffer_from_pool called before initialization.\n");
         if (actual_size) *actual_size = 0;
         return NULL;
    }

    // Acquire mutex to safely access pool state
    pthread_mutex_lock(&pool_mutex);

    starting_idx = next_buffer_idx; // Start searching from the next expected index

    // Search for a free buffer slice in a cyclic manner
    for (int i = 0; i < POOL_BUFFER_COUNT; ++i) {
        int current_idx = (starting_idx + i) % POOL_BUFFER_COUNT;

        if (buffer_status[current_idx] == 1) { // Found a free buffer slice
            // Calculate the pointer to the start of this buffer slice
            buffer = base_memory_block + (size_t)current_idx * POOL_BUFFER_SIZE; // Use size_t cast for arithmetic safety
            buffer_status[current_idx] = 0; // Mark as in use
            next_buffer_idx = (current_idx + 1) % POOL_BUFFER_COUNT; // Update next index
            if (actual_size) *actual_size = POOL_BUFFER_SIZE; // Return the buffer slice size
            // fprintf(stderr, "DEBUG: Get buffer index %d -> %p\n", current_idx, buffer);
            break; // Exit the loop, buffer found
        }
    }

    // Release mutex
    pthread_mutex_unlock(&pool_mutex);

    if (buffer == NULL) {
         // fprintf(stderr, "DEBUG: Pool exhausted, get returned NULL\n");
         if (actual_size) *actual_size = 0; // Indicate 0 size buffer if NULL
    }

    return buffer; // Returns buffer pointer or NULL if pool is full
}

// Return a buffer slice back to the pool, marking it as free.
void return_buffer_to_pool(uint8_t* buffer) {
    int found_idx = -1;

    if (!pool_initialized) {
         fprintf(stderr, "Error: return_buffer_to_pool called before initialization.\n");
         return;
    }

    // Keep this if we want to allocate the buffer on the heap
    if (buffer == NULL) {
        // Silently ignore returning NULL, or log a warning
        // fprintf(stderr, "Warning: return_buffer_to_pool called with NULL buffer.\n");
        return;
    }

    pthread_mutex_lock(&pool_mutex);

    // Validate the pointer and calculate its index
    // Use ptrdiff_t for pointer difference
    ptrdiff_t offset = buffer - base_memory_block;

    // Check if the pointer is within the managed block and is correctly aligned
    if (buffer >= base_memory_block &&
        buffer < base_memory_block + TOTAL_POOL_SIZE &&
        offset >= 0 && // Offset must be non-negative
        (size_t)offset % POOL_BUFFER_SIZE == 0) // Check alignment
    {
        // Pointer is valid and aligned, calculate the index
        found_idx = offset / POOL_BUFFER_SIZE;

        // Double-check index is within bounds (should be if pointer checks pass, but defensive)
        if (found_idx >= 0 && found_idx < POOL_BUFFER_COUNT) {
             if (buffer_status[found_idx] == 0) { // Check if it was marked as in use
                  buffer_status[found_idx] = 1; // Mark as free
                  // fprintf(stderr, "DEBUG: Returned buffer index %d <- %p\n", found_idx, buffer);
             } else {
                  // Indicates returning a buffer already marked as free (potential double return or logic error)
                  fprintf(stderr, "Warning: Attempted to return buffer %p at index %d which was already free.\n", buffer, found_idx);
             }
        } else {
            // This case should ideally not be reachable if the initial pointer checks are correct
            fprintf(stderr, "Error: Calculated index %d out of bounds for buffer %p.\n", found_idx, buffer);
        }
    } else {
        // The buffer pointer is not within the managed memory block or is misaligned
        fprintf(stderr, "Error: Attempted to return a buffer %p not belonging to the pool's memory block %p or misaligned.\n", buffer, base_memory_block);
        // exit(EXIT_FAILURE);
    }

    // Release mutex
    pthread_mutex_unlock(&pool_mutex);
}
