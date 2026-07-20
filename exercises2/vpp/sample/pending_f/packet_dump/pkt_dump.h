#ifndef __included_ck_sample_h__
#define __included_ck_sample_h__

#include <vnet/vnet.h>
#include <vnet/ip/ip.h>

#include <vppinfra/hash.h>
#include <vppinfra/error.h>
#include <vppinfra/elog.h>

typedef struct {
    /* API message ID base */
    u16 msg_id_base;

    /* convenience */
    vnet_main_t * vnet_main;
} ck_sample_main_t;

extern ck_sample_main_t ck_sample_main;

extern vlib_node_registration_t ck_sample_node;

#define CK_SAMPLE_PLUGIN_BUILD_VER "1.0"

#endif /* __included_ck_sample_h__ */