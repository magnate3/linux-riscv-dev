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
#ifndef __included_learn_h__
#define __included_learn_h__

#include <vnet/vnet.h>
#include <vnet/ip/ip.h>
#include <vnet/ethernet/ethernet.h>

#include <vppinfra/hash.h>
#include <vppinfra/error.h>
#include <vppinfra/elog.h>

// 0xff & _addr >> 24; 0xff & _addr >> 16;
// 0xff & _addr >> 8; 0xff & _addr

#define learn_elog_addr(_str, _addr)                 \
do                                                   \
  {                                                  \
    ELOG_TYPE_DECLARE (e) =                          \
      {                                              \
        .format = _str " %d.%d.%d.%d",               \
        .format_args = "i1i1i1i1",                   \
      };                                             \
    CLIB_PACKED(struct                               \
      {                                              \
        u8 oct1;                                     \
        u8 oct2;                                     \
        u8 oct3;                                     \
        u8 oct4;                                     \
      }) *ed;                                        \
    ed = ELOG_DATA (&vlib_global_main.elog_main, e); \
    ed->oct4 = _addr >> 24;                          \
    ed->oct3 = _addr >> 16;                          \
    ed->oct2 = _addr >> 8;                           \
    ed->oct1 = _addr;                                \
  } while (0);

/*static inline u32
elog_id_for_msg_name (vlib_main_t * vm, const char *msg_name)
{
  uword *p, r;
  static uword *h;
  u8 *name_copy;

  if (!h)
    h = hash_create_string (0, sizeof (uword));

  p = hash_get_mem (h, msg_name);
  if (p)
    return p[0];
  r = elog_string (&vm->elog_main, "%s", msg_name);

  name_copy = format (0, "%s%c", msg_name, 0);

  hash_set_mem (h, name_copy, r);

  return r;
}*/

/*#define nat_elog(nat_elog_sev, nat_elog_str)                        \
do                                                                  \
  {                                                                 \
    snat_main_t *sm = &snat_main;                                   \
    if (PREDICT_FALSE (sm->log_level >= SNAT_LOG_INFO))             \
      {                                                             \
        ELOG_TYPE_DECLARE (e) =                                     \
          {                                                         \
            .format = "nat-msg: (info) %s",                         \
            .format_args = "T4",                                    \
          };                                                        \
        CLIB_PACKED(struct                                          \
        {                                                           \
          u32 c;                                                    \
        }) *ed;                                                     \
        ed = ELOG_DATA (&sm->vlib_main->elog_main, e);              \
        ed->c = elog_id_for_msg_name (sm->vlib_main, nat_elog_str); \
      }                                                             \
  } while (0);*/

typedef struct {
    /* API message ID base */
    u16 msg_id_base;

    /* convenience */
    vnet_main_t * vnet_main;
} learn_main_t;

extern learn_main_t learn_main;

extern vlib_node_registration_t learn_node;

#define LEARN_PLUGIN_BUILD_VER "1.0"

#endif /* __included_learn_h__ */
