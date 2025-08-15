#ifndef __DPDK_REASSEMBLY__
#define __DPDK_REASSEMBLY__
#include <rte_ip_frag.h>
#define RTE_LOGTYPE_IP_RSMBL RTE_LOGTYPE_USER1
/* Configure how many packets ahead to prefetch, when reading packets */
#define PREFETCH_OFFSET 3
#define MS_PER_S 1000
#define MAX_FLOW_NUM    UINT16_MAX
#define MIN_FLOW_NUM    1
#define DEF_FLOW_NUM    0x1000

/* TTL numbers are in ms. */
#define MAX_FLOW_TTL    (3600 * MS_PER_S)
#define MIN_FLOW_TTL    1
#define DEF_FLOW_TTL    MS_PER_S

#define MAX_FRAG_NUM RTE_LIBRTE_IP_FRAG_MAX_FRAG

/* Should be power of two. */
#define IP_FRAG_TBL_BUCKET_ENTRIES      16

extern uint32_t max_flow_num ;
extern uint32_t max_flow_ttl ;
#endif
