#include <vlib/vlib.h>
#include <vnet/plugin/plugin.h>
#include <vpp/app/version.h>
#include <plugins/my_node/my_ip.h>
#include <vnet/feature/feature.h>


#if 1
VLIB_PLUGIN_REGISTER () = {
	  .version = MY_IP_BUILD_VER,
	  .description = "my_ip",
};
#endif
