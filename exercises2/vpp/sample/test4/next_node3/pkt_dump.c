#include <vnet/plugin/plugin.h>
#include <packet_dump/pkt_dump.h>

ck_sample_main_t ck_sample_main;


int ck_sample_enable_disable(u32 sw_if_index, int enable)
{
        if (pool_is_free_index (ck_sample_main.vnet_main->interface_main.sw_interfaces, 
                                sw_if_index))
                return VNET_API_ERROR_INVALID_SW_IF_INDEX;

        vnet_feature_enable_disable("ip4-unicast",
                "ck_sample",
                sw_if_index, enable, 0, 0);
        return 0;
}


static clib_error_t*
ck_sample_enable_disable_command_fn(vlib_main_t* vm,
                                    unformat_input_t *input,
                                    vlib_cli_command_t *cmd)
{
        u32 sw_if_index = ~0;
        int enable_disable = 1;

        while(unformat_check_input(input) != UNFORMAT_END_OF_INPUT) {
                if (unformat(input, "disable"))
                        enable_disable = 0;
                else if (unformat(input, "%U",
                        unformat_vnet_sw_interface,
                        ck_sample_main.vnet_main, &sw_if_index));
                else
                        break;
        }

        if (sw_if_index == ~0)
                return clib_error_return(0, "Please specify an interface...");

        ck_sample_enable_disable(sw_if_index, enable_disable);

        return 0;
}

VLIB_CLI_COMMAND (ck_sample_command, static) = {
    .path = "pkt dump",
    .short_help = 
    "pkt dump <interface-name> [disable]",
    .function = ck_sample_enable_disable_command_fn,
};


VLIB_PLUGIN_REGISTER () = {
    .version = CK_SAMPLE_PLUGIN_BUILD_VER,
    .description = "Sample of VPP Plugin",
};

#if 1
static clib_error_t *sample_init(vlib_main_t* vm)
{
     return 0;
}
VNET_FEATURE_INIT(ck_sample, static) = 
{
	.arc_name = "ip4-unicast",
	.node_name = "ck_sample",
	.runs_before = VNET_FEATURES("ip4-lookup"),
};
VLIB_INIT_FUNCTION(sample_init);
#else
static clib_error_t *sample_init(vlib_main_t* vm)
{
     vlib_node_t *node, *next_node;
     //vlib_node_t *node;
     next_node = vlib_get_node_by_name (vm, (u8 *) "ip4-lookup");
     node = vlib_get_node_by_name (vm, (u8 *) "ck_sample");
     if (node && next_node)
     {
        vlib_cli_output (vm, "%s run  find ip4-lookup and ck_sample node \n",__func__);
         
	vlib_node_add_next(vm,node->index, next_node->index);  
     }
     //vlib_cli_output (vm, "%s run  %s \n", __func__, node ? "find ip4-lookup node": "not find ip4-lookup node");
     return 0;
}
VLIB_INIT_FUNCTION(sample_init);
#endif
static clib_error_t *ck_sample_init(vlib_main_t* vm)
{
	ck_sample_main.vnet_main = vnet_get_main();
	vlib_call_init_function (vm, sample_init);
	return 0;
}

VLIB_INIT_FUNCTION(ck_sample_init);

