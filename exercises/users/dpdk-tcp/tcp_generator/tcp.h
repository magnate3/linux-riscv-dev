#ifndef __TCP__
#define __TCP__

//#include <rte_common.h>
//#include <rte_ether.h>
//#include <assert.h>
//#include <rte_ip.h>
#include <rte_tcp.h>
#include "tcp_misc.h"

/*
 *  * TCP protocols related structures/functions definitions.
 *   * Main purpose to simplify (and optimise) processing and representation
 *    * of protocol related data.
 *     */

#define	TCP_WSCALE_DEFAULT	7
#define	TCP_WSCALE_NONE		0

#define	TCP_TX_HDR_MAX	(sizeof(struct rte_tcp_hdr) + TCP_TX_OPT_LEN_MAX)

/* max header size for normal data+ack packet */
#define	TCP_TX_HDR_DACK	(sizeof(struct rte_tcp_hdr) + TCP_TX_OPT_LEN_TMS)

#define	TCP4_MIN_MSS	536

#define	TCP6_MIN_MSS	1220

/* default MTU, no TCP options. */
#define TCP4_NOP_MSS	\
	(RTE_ETHER_MTU - sizeof(struct rte_ipv4_hdr) - \
	 sizeof(struct rte_tcp_hdr))

#define TCP6_NOP_MSS	\
	(RTE_ETHER_MTU - sizeof(struct rte_ipv6_hdr) - \
	 sizeof(struct rte_tcp_hdr))

/* default MTU, TCP options present */
#define TCP4_OP_MSS	(TCP4_NOP_MSS - TCP_TX_OPT_LEN_MAX)

#define TCP6_OP_MSS	(TCP6_NOP_MSS - TCP_TX_OPT_LEN_MAX)
#endif
