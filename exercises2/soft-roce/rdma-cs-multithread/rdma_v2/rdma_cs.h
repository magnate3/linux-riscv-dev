/**
 * @file rdma_cs.h
 * @author Austin Pohlmann
 * @brief The header file containing resources used in both the client and the server
 */
#ifndef RDMA_CS_HEADER
#define RDMA_CS_HEADER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#include <infiniband/verbs.h>
#include <infiniband/arch.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
/**
 * @brief The max amount of send type work requests
 */
#define MAX_SEND_WR		8
/**
 * @brief The max amount of send type scatter/gather elements
 */
#define MAX_SEND_SGE	4
/**
 * @brief The max amount of receive type work requests
 */
#define MAX_RECV_WR		8
/**
 * @brief The max amount of receive type scatter/gather elements
 */
#define MAX_RECV_SGE	4
/**
 * @brief The max amount (in bytes) that can be written inline
 */
#define MAX_INLINE_DATA	256
/**
 * @brief The default local memory region size
 */
#define REGION_LENGTH	512
/**
 * @brief The default memory region size on the server
 */
#define SERVER_MR_SIZE	1024
/**
 * @brief The file path to store the server logs to
 */
#define SERVER_LOG_PATH	"./server_logs/"

/**
 * @brief Determines the type of completion is being fetched
 */
enum completion_type {
	RECV,	/**< Used for receive operations */
	SEND	/**< Used for send, rdma read, and rdma write operations (only for SIGNLAED rdma operations) */
};

/**
 * @brief Standard opcodes for operations done between hosts
 */
enum client_opcodes {
	DISCONNECT = 1,	/**< Send a disconnect request to the remote host */
	WRITE_INLINE,	/**< Perform an inline rdma write */
	WRITE,			/**< Perform an rdma write */
	READ,			/**< Perform and rdma read */
	OPEN_MR,		/**< Open a memory region on the server */
	CLOSE_MR,		/**< Close a memory region on the server */
	ADD_CLIENT = 10,/**< Used to add an open memory regions to clients' lists */
	REMOVE_CLIENT	/**< Used to remove open memory regions from clients' lists */
};

/**
 * @brief Node for a linked lists containing information about open memory regions
 */
struct client {
	unsigned long cid;		/**< The numerical identification number of the client that owns the memory region */
	uint32_t rkey;			/**< The rkey associated with the memory region */
	uint64_t remote_addr;	/**< The address on the server of the memory region */
	size_t length;			/**< The length of the memory region */
	struct client *next;	/**< A pointer to the next node in the list */
};

uint32_t get_completion(struct rdma_cm_id *, enum completion_type, uint8_t, FILE *);
struct rdma_cm_id *cm_event(struct rdma_event_channel *, enum rdma_cm_event_type, FILE *);
void swap_info(struct rdma_cm_id *, struct ibv_mr *, uint32_t *, uint64_t *, size_t *, FILE *);
int obliterate(struct rdma_cm_id *,struct rdma_cm_id *, struct ibv_mr *, struct rdma_event_channel *, FILE *);
void stop_it(char *, int, FILE *);
void rdma_recv(struct rdma_cm_id *, struct ibv_mr *, FILE *);
void rdma_send_op(struct rdma_cm_id *, uint8_t, FILE *);
void rdma_write_inline(struct rdma_cm_id *, void *, uint64_t, uint32_t, FILE *);
#endif