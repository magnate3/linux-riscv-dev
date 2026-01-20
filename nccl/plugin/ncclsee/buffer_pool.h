#ifndef BUFFER_POOL_H_
#define BUFFER_POOL_H_


#include <stdint.h>
#include <stddef.h>

// --------------------------------------------------------------------------
// Buffer Pool Configuration
// --------------------------------------------------------------------------

// Size of each buffer chunk within the pool
#define POOL_BUFFER_SIZE (16384)

// Number of buffer chunks in the pool.
#define POOL_BUFFER_COUNT 64

// Total size of the contiguous memory block required for the entire pool
#define TOTAL_POOL_SIZE (POOL_BUFFER_SIZE * POOL_BUFFER_COUNT)

// --------------------------------------------------------------------------
// Buffer Pool Management Functions
// These functions manage pointers pointing into a pre-allocated memory block.
// The memory block itself is NOT allocated or freed by these functions;
// it must be provided by the caller during initialization.
// --------------------------------------------------------------------------

// Initialize the buffer pool.
// It configures the pool to manage the provided contiguous memory_block.
// The memory_block must be of size TOTAL_POOL_SIZE.
//
// memory_block: Pointer to the start of the memory block to be managed.
void init_buffer_pool(uint8_t* memory_block);

// Destroy the buffer pool.
// Cleans up any internal pool resources (like mutexes).
// It DOES NOT free the memory_block provided to init_buffer_pool.
void destroy_buffer_pool(void);

// Get a buffer (a pointer to a POOL_BUFFER_SIZE chunk within the memory block)
// from the pool. It uses a cyclic approach to find the next available buffer.
// Returns NULL if no buffer is available (pool is exhausted).
//
// actual_size [out]: Pointer to a size_t where the actual size of the
//                    returned buffer (which is POOL_BUFFER_SIZE) will be stored.
//                    This pointer can be NULL if the caller doesn't need the size.
uint8_t* get_buffer_from_pool(size_t* actual_size);

// Return a buffer (a pointer previously obtained from get_buffer_from_pool)
// back to the pool, marking its corresponding chunk as free.
// Includes validation to ensure the pointer belongs to the managed block.
//
// buffer [in]: The buffer pointer to return.
void return_buffer_to_pool(uint8_t* buffer);

#endif // BUFFER_POOL_H_
