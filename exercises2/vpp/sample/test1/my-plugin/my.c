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
#define TEST_INTERFACE_INDEX 1
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
#if 0
    u32 sw_if_index = ~0;
    my_main_t * sm = &my_main;
    int enable_disable = 1;
     while (unformat_check_input (input) != UNFORMAT_END_OF_INPUT) {
     if (unformat (input, "disable"))
	       enable_disable = 0;
     else if (unformat (input, "%U", unformat_vnet_sw_interface,
                           sm->vnet_main, &sw_if_index))
       ;
     else
       break;
   }

   if (sw_if_index == ~0)
	           return clib_error_return (0, "Please specify an interface...");
   vlib_cli_output (vm, "my cmd run interface index %u and enable %d  \n", sw_if_index, enable_disable);
#else
   vlib_cli_output (vm, "my cmd run interface index %u and enable \n", TEST_INTERFACE_INDEX);
   //vnet_feature_enable_disable ("device-input", "my", TEST_INTERFACE_INDEX, 1, NULL, 0);
   //vnet_feature_enable_disable ("arp", "my_ip", TEST_INTERFACE_INDEX, 1, NULL, 0);
   vnet_feature_enable_disable ("ip4-unicast", "my_ip", TEST_INTERFACE_INDEX, 1, NULL, 0);
   //vnet_feature_enable_disable ("device-input", "my_ip", TEST_INTERFACE_INDEX, 1, NULL, 0);
#endif
   //vnet_feature_enable_disable (arc_name, node_name, sw_if_index, enable_disable, feature_config, n_feature_config_bytes);
  return 0;
}
/**
 * @brief CLI command to enable/disable the my macswap plugin.
 */
VLIB_CLI_COMMAND (mine_command, static) = {
    .path = "mine",
    //.path = "my",
    .short_help = 
    "my feat <interface-name> [disable]",
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
#if 1
VNET_FEATURE_INIT (my, static) = 
{
  .arc_name = "device-input",
  .node_name = "my",
  .runs_before = VNET_FEATURES ("ethernet-input"),
};
#endif
