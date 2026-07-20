#pragma once

#include <linux/types.h>
#define __unused __attribute__ ((unused))

#define LIKELY(x) (__builtin_expect (!!(x), 1))
#define UNLIKELY(x) (__builtin_expect (!!(x), 0))

#define BARRIER() __asm__ __volatile__ ("" ::: "memory")

#define BUSY_WAIT(cond) \
    do                  \
    {                   \
        BARRIER ();     \
    } while (cond)

#ifndef DEBUG
#define DEBUG 0
#endif

/**
 * SERVER is specified at compilation time by CMake.
 */
#ifndef SERVER
#define SERVER 0
#endif

#if DEBUG
#define LOG(stream, fmt, ...)                 \
    do                                        \
    {                                         \
        fprintf (stream, fmt, ##__VA_ARGS__); \
        fflush (stream);                      \
    } while (0)

#define PERROR(errno)   \
    do                  \
    {                   \
        perror (errno); \
    } while (0)
#else
#define LOG(stream, fmt, ...)
#define PERROR(errno)
#endif

#define START_TIMER() uint64_t __start = get_time_ns ()
#define STOP_TIMER() uint64_t __end = get_time_ns ()
#define STOP_TIMER_AND_PRINT()                   \
    do                                           \
    {                                            \
        uint64_t __end = get_time_ns ();         \
        printf ("Time: %lu\n", __end - __start); \
        fflush (stdout);                         \
    } while (0)

#define TIME_CODE(code)                      \
    do                                       \
    {                                        \
        uint64_t start = get_time_ns ();     \
        (code);                              \
        uint64_t end = get_time_ns ();       \
        printf ("Time: %lu\n", end - start); \
        fflush (stdout);                     \
    } while (0)

/**
 * Size of the whole pingpong packet.
 * Although the actual payload (Ethernet header + IP header + pingpong payload) is smaller than this,
 * the packet should be the same across all the XDP, RDMA and DPDK measurements.
 */
#define PACKET_SIZE 1024

/**
 * Size of the exchanged packet containing MAC address (6 bytes) and IP address (4 bytes) of each machine.
 * This packet is sent before the start of pingpong to exchange the addresses without hardcoding them
 */
#define ETH_IP_INFO_PACKET_SIZE (ETH_ALEN + sizeof (uint32_t))

#define ROCE_INFO_PACKET_SIZE (sizeof (int) * 3 + 16)// LID (int), QPN (int), PSN (int), GID (16 bytes)

// Custom ethernet protocol number
#define ETH_P_PINGPONG 0x2002

// UDP port used for the pingpong communication in pp_pure.c
#define XDP_UDP_PORT 1234

// Size of the map used to exchange the pingpong packets
// The bigger, the slower the polling but the less likely to lose packets
#define PACKETS_MAP_SIZE 128

// Random magic number for pingpong packets
#define PINGPONG_MAGIC 0x8badbeef


#pragma pack (push, 1)
struct pingpong_payload {
    __u64 id;
    __u64 ts[4];

    __u32 phase;
    /**
     * In an unsynchronized XDP-userspace polling communication, there is the possibility of a corruption of packets in the case of XDP writing the same space in memory that userspace is reading.
     * To address this issue, a "magic number" was added at the end of the pingpong payload, which helps recognize the integrity of the packet without need of any checksum: before reading a packets from the map,
     * the userspace app verified that the magic number matches; after reading a packet, its memory space in the eBPF map is zeroed.
     * By doing this, the only possible way to read a packet in userspace is for XDP to have completely written the packet, including the magic number which is placed at the end;
     * in the case of a half-written packet, the magic number would not match stopping the userspace from reading it.
     */
    __u32 magic;
};
#pragma pack (pop)

inline struct pingpong_payload empty_pingpong_payload ()
{
    struct pingpong_payload payload;
    payload.id = 0;
    payload.phase = 0;
    payload.ts[0] = 0;
    payload.ts[1] = 0;
    payload.ts[2] = 0;
    payload.ts[3] = 0;
    payload.magic = PINGPONG_MAGIC;

    return payload;
}

inline struct pingpong_payload new_pingpong_payload (__u64 id)
{
    struct pingpong_payload payload;
    payload.id = id;
    payload.phase = 0;
    payload.ts[0] = 0;
    payload.ts[1] = 0;
    payload.ts[2] = 0;
    payload.ts[3] = 0;
    payload.magic = PINGPONG_MAGIC;

    return payload;
}

/**
 * Check if the given payload is valid.
 * A payload is valid if its magic number is equal to PINGPONG_MAGIC.
 *
 * @param payload the payload to check
 * @return 1 if the payload is valid, 0 otherwise
 */
inline __u32 valid_pingpong_payload (const volatile struct pingpong_payload *volatile payload)
{
    return payload->magic == PINGPONG_MAGIC;
}

inline __u64 compute_latency (const struct pingpong_payload *payload)
{
    return ((payload->ts[3] - payload->ts[0]) - (payload->ts[2] - payload->ts[1])) / 2;
}