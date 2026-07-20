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
#include <plugins/my-plugin/my.h>

#include <vlibapi/api.h>
#include <vlibmemory/api.h>

#include <plugins/my-plugin/my.api_enum.h>
#include <plugins/my-plugin/my.api_types.h>

#define REPLY_MSG_ID_BASE sm->msg_id_base
#include <vlibapi/api_helper_macros.h>

/* *INDENT-OFF* */
VLIB_PLUGIN_REGISTER () = {
    .version = MY_PLUGIN_BUILD_VER,
    .description = "my_mine of VPP Plugin",
};
/* *INDENT-ON* */

my_main_t my_main;
static clib_error_t *
my_command_fn (vlib_main_t * vm,
                                   unformat_input_t * input,
                                   vlib_cli_command_t * cmd)
{
   vlib_cli_output (vm, "mine cmd run \n");
  return 0;
}
/**
 * @brief CLI command to enable/disable the my macswap plugin.
 */
VLIB_CLI_COMMAND (mine_command, static) = {
    .path = "mine",
    .short_help = 
    "my macswap <interface-name> [disable]",
    .function = my_command_fn,
};

#if 1
/**
 * @brief Plugin API message handler.
 * vl_api_my_response_t_handler
 */
static void vl_api_my_response_t_handler
(vl_api_my_response_t * mp)
{
#if 0
  vl_api_my_macswap_enable_disable_reply_t * rmp;
  my_main_t * sm = &my_main;
  int rv;
#endif
  
  //REPLY_MACRO(VL_API_SAMPLE_MACSWAP_ENABLE_DISABLE_REPLY);
}
#endif
/* API definitions */
#include <plugins/my-plugin/my.api.c>

/**
 * @brief Initialize the my plugin.
 */
static clib_error_t * my_init (vlib_main_t * vm)
{
  my_main_t * sm = &my_main;

  sm->vnet_main =  vnet_get_main ();

  /* Add our API messages to the global name_crc hash table */
  sm->msg_id_base = setup_message_id_table ();

  return 0;
}

VLIB_INIT_FUNCTION (my_init);

/**
 * @brief Hook the my plugin into the VPP graph hierarchy.
 */
VNET_FEATURE_INIT (my, static) = 
{
  .arc_name = "device-input",
  .node_name = "my",
  .runs_before = VNET_FEATURES ("ethernet-input"),
};
