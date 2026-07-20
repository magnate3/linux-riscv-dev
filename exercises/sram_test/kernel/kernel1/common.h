#ifndef COMMON_H
#define COMMON_H
typedef struct mutex mutex_t;
#define __STANDALONE__
#if !defined(__STANDALONE__) && defined(__KERNEL__)
  mutex_t gSramMutex;        ///< SRAM locking mutex
  // exported to DL allocator
  mutex_t magic_init_mutex;  ///< MAGIC mutex
  mutex_t mspace_mutex;      ///< MAGIC mutex
  mutex_t gMoreCoreMutex;    ///< more core mutex
#else // #if !defined(__STANDALONE__) && defined(__KERNEL__)
  // placeholders
  uint32_t gSramMutex;
  // exported to DL allocator
  uint32_t magic_init_mutex;
  uint32_t mspace_mutex;
  uint32_t gMoreCoreMutex;
#endif 
#endif
