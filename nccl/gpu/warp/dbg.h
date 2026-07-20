#ifndef __dbg_h__
#define __dbg_h__

#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef NDEBUG
#    define debug(M, ...)
#else
#    define debug(M, ...)                                                              \
        fprintf(stderr, "[DEBUG] %s:%s:%d: " M "\n", __FUNCTION__, __FILE__, __LINE__, \
                ##__VA_ARGS__)
#endif

#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_ORANGE "\x1b[33m"
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define clean_errno() (errno == 0 ? "None" : strerror(errno))

#define log_err(M, ...)                                                                       \
    fprintf(stderr, ANSI_COLOR_RED "[ERROR]\x1b[0m %s:%s:%d errno: %s " M "\n", __FUNCTION__, \
            __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)

#define log_warn(M, ...)                                                                          \
    fprintf(stderr, ANSI_COLOR_ORANGE "[WARN]\x1b[0m \t%s:%s:%d errno: %s " M "\n", __FUNCTION__, \
            __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)

#define log_info(M, ...)                                                                         \
    fprintf(stderr, ANSI_COLOR_GREEN "[INFO]\x1b[0m \t%s:%s:%d " M "\n", __FUNCTION__, __FILE__, \
            __LINE__, ##__VA_ARGS__)

#define check(A, M, ...)           \
    if (!(A))                      \
    {                              \
        log_err(M, ##__VA_ARGS__); \
        errno = 0;                 \
        goto error;                \
    }

#define sentinel(M, ...)           \
    {                              \
        log_err(M, ##__VA_ARGS__); \
        errno = 0;                 \
        goto error;                \
    }

#define check_mem(A) check((A), "Out of memory.")

#define check_debug(A, M, ...)   \
    if (!(A))                    \
    {                            \
        debug(M, ##__VA_ARGS__); \
        errno = 0;               \
        goto error;              \
    }

#define CHECK(A)                                                 \
    {                                                            \
        const cudaError_t status = A;                            \
        if (status != cudaSuccess)                               \
        {                                                        \
            log_err("reason: %s\n", cudaGetErrorString(status)); \
            cudaDeviceReset();                                   \
            exit(1);                                             \
        }                                                        \
    }

#endif
