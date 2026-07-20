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

static clib_error_t *ck_sample_init(vlib_main_t* vm)
{
	ck_sample_main.vnet_main = vnet_get_main();
	return 0;
}

VLIB_INIT_FUNCTION(ck_sample_init);

VNET_FEATURE_INIT(ck_sample, static) = 
{
	.arc_name = "ip4-unicast",
	.node_name = "ck_sample",
	.runs_before = VNET_FEATURES("ip4-lookup"),
};
