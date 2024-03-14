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
/**
 * @file
 * @brief Sample Plugin, plugin API / trace / CLI handling.
 */

#include <vnet/vnet.h>
#include <vnet/plugin/plugin.h>
#include <learn-vpp/learn.h>

#include <vlibapi/api.h>
#include <vlibmemory/api.h>

/* define message IDs */
#include <learn-vpp/learn_msg_enum.h>
/* define message structures */
#define vl_typedefs
#include <learn-vpp/learn_all_api_h.h> 
#undef vl_typedefs

/* define generated endian-swappers */
#define vl_endianfun
#include <learn-vpp/learn_all_api_h.h> 
#undef vl_endianfun

/* instantiate all the print functions we know about */
#define vl_print(handle, ...) vlib_cli_output (handle, __VA_ARGS__)
#define vl_printfun
#include <learn-vpp/learn_all_api_h.h> 
#undef vl_printfun

/* Get the API version number */
#define vl_api_version(n,v) static u32 api_version=(v);
#include <learn-vpp/learn_all_api_h.h>
#undef vl_api_version

#define REPLY_MSG_ID_BASE sm->msg_id_base
#include <vlibapi/api_helper_macros.h>

/* List of message types that this plugin understands */

#define foreach_learn_plugin_api_msg                           \
_(LEARN_ENABLE_DISABLE, learn_enable_disable)

/* *INDENT-OFF* */
VLIB_PLUGIN_REGISTER () = {
    .version = LEARN_PLUGIN_BUILD_VER,
    .description = "Learn VPP Plugin",
};
/* *INDENT-ON* */

learn_main_t learn_main;

/**
 * @brief Enable/disable the learn plugin. 
 *
 * Action function shared between message handler and debug CLI.
 */
#define TEST_INTERFACE_INDEX 1
int learn_enable_disable (learn_main_t * sm, int enable_disable)
{
  int rv = 0;



  vnet_feature_enable_disable ("ip4-unicast", "learn-vpp-input",
                                TEST_INTERFACE_INDEX, enable_disable, 0, 0);

  return rv;
}

static clib_error_t *
learn_enable_disable_command_fn (vlib_main_t * vm,
                                   unformat_input_t * input,
                                   vlib_cli_command_t * cmd)
{
  learn_main_t * sm = &learn_main;
  int enable_disable = 1;
    
  int rv;

  while (unformat_check_input (input) != UNFORMAT_END_OF_INPUT) {
    if (unformat (input, "disable"))
      enable_disable = 0;
    else
      break;
  }

  rv = learn_enable_disable (sm, enable_disable);

  switch(rv) {
  case 0:
    break;

  default:
    return clib_error_return (0, "learn_enable_disable returned %d",
                              rv);
  }
  return 0;
}

/**
 * @brief CLI command to enable/disable the learn plugin.
 */
VLIB_CLI_COMMAND (sr_content_command, static) = {
    .path = "learn-vpp",
    .short_help = 
    "learn-vpp [disable]",
    .function = learn_enable_disable_command_fn,
};

/**
 * @brief Plugin API message handler.
 */
static void vl_api_learn_enable_disable_t_handler
(vl_api_learn_enable_disable_t * mp)
{
  vl_api_learn_enable_disable_reply_t * rmp;
  learn_main_t * sm = &learn_main;
  int rv;

  rv = learn_enable_disable (sm, (int) (mp->enable_disable));
  
  REPLY_MACRO(VL_API_LEARN_ENABLE_DISABLE_REPLY);
}

/**
 * @brief Set up the API message handling tables.
 */
static clib_error_t *
learn_plugin_api_hookup (vlib_main_t *vm)
{
  learn_main_t * sm = &learn_main;
#define _(N,n)                                                  \
    vl_msg_api_set_handlers((VL_API_##N + sm->msg_id_base),     \
                           #n,					\
                           vl_api_##n##_t_handler,              \
                           vl_noop_handler,                     \
                           vl_api_##n##_t_endian,               \
                           vl_api_##n##_t_print,                \
                           sizeof(vl_api_##n##_t), 1); 
    foreach_learn_plugin_api_msg;
#undef _

    return 0;
}

/**
 * @brief Initialize the learn plugin.
 */
//#include <vlibapi/api_helper_macros.h>
#if 1
//#include <plugins/learn-vpp/learn.api.c>
#define vl_msg_name_crc_list
#include <learn-vpp/learn_all_api_h.h>
#undef vl_msg_name_crc_list
static void 
setup_message_id_table (learn_main_t * sm, api_main_t *am)
{
#define _(id,n,crc) \
  vl_msg_api_add_msg_name_crc (am, #n "_" #crc, id + sm->msg_id_base);
  foreach_vl_msg_name_crc_learn;
#undef _
}
static clib_error_t * learn_init (vlib_main_t * vm)
{
  learn_main_t * sm = &learn_main;
  api_main_t *am = vlibapi_get_main ();
  clib_error_t * error = 0;
  u8 * name;

  sm->vnet_main =  vnet_get_main ();

  name = format (0, "learn-vpp_%08x%c", api_version, 0);

  /* Ask for a correctly-sized block of API message decode slots */
  sm->msg_id_base = vl_msg_api_get_msg_ids 
      ((char *) name, VL_MSG_FIRST_AVAILABLE);

  error = learn_plugin_api_hookup (vm);

  /* Add our API messages to the global name_crc hash table */
  setup_message_id_table (sm, am);
  vec_free(name);

  return error;
}
#else
//#include <plugins/learn-vpp/learn.api.c>
//#include <plugins/learn-vpp/learn.api_types.h>
#include <plugins/learn-vpp/learn.api_enum.h>
static clib_error_t * learn_init (vlib_main_t * vm)
{
  learn_main_t * sm = &learn_main;
  clib_error_t * error = 0;
  u8 * name;

  sm->vnet_main =  vnet_get_main ();

  name = format (0, "learn-vpp_%08x%c", api_version, 0);


  error = learn_plugin_api_hookup (vm);

  /* Add our API messages to the global name_crc hash table */
  //setup_message_id_table (sm, &api_main);
  sm->msg_id_base = setup_message_id_table ();
  vec_free(name);

  return error;
}
#endif
VLIB_INIT_FUNCTION (learn_init);

/**
 * @brief Hook the learn plugin into the VPP graph hierarchy.
 */
VNET_FEATURE_INIT (learn, static) = 
{
  .arc_name = "ip4-unicast",
  .node_name = "learn-vpp-input",
  .runs_before = VNET_FEATURES ("acl-plugin-in-ip4-fa"),
};
