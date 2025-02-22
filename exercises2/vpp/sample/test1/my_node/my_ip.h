/*
 * Copyright (c) 2015 Cisco and/or its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __included_my_ip_h__
#define __included_my_ip_h__

#include <vnet/vnet.h>
#include <vnet/ip/ip.h>
#include <vnet/ethernet/ethernet.h>

#include <vppinfra/hash.h>
#include <vppinfra/error.h>
#include <vppinfra/elog.h>
#define  MY_IP_BUILD_VER "1.0"
typedef struct {
    /* API message ID base */
    u16 msg_id_base;

    u32 frame_queue_index;
    /* convenience */
    vnet_main_t * vnet_main;
} my_ip_main_t;

extern my_ip_main_t my_ip_main;

extern vlib_node_registration_t my_ip_node;

#define MY_PLUGIN_BUILD_VER "1.0"

#endif /* __included_my_ip_h__ */
