

#  ip node

![image](../../pic/my.png)

## make build

```
   cp ./vpp/build-root/build-vpp_debug-native/vpp/lib/vpp_plugins/my_plugin.so   ./vpp/build-root/build-vpp-native/vpp/lib/vpp_plugins
 cp ./vpp/build-root/build-vpp_debug-native/vpp/lib/vpp_plugins/my_ip_plugin.so   ./vpp/build-root/build-vpp-native/vpp/lib/vpp_plugins
```

## feature

```C
VNET_FEATURE_INIT (my_ip_node, static) =
{

  //.arc_name = "device-input",
  .arc_name = "ip4-unicast",
  .node_name = "my_ip",
  .runs_before = VNET_FEATURES ("ip4-lookup"),
  //.start_nodes = VNET_FEATURES ("ip4-input"),  
  //.node_name = "interface-output",
  //.runs_before = VNET_FEATURES ("ethernet-input"),
};
```
### enable feature
```C
my_command_fn (vlib_main_t * vm,
                                   unformat_input_t * input,
                                   vlib_cli_command_t * cmd)
{
 
   vnet_feature_enable_disable ("ip4-unicast", "my_ip", TEST_INTERFACE_INDEX, 1, NULL, 0);
   
}
```

##  next_nodes
```C
VLIB_REGISTER_NODE (my_ip_node) =
{
  .name = "my_ip",
  .function = my_ip_node_fn,
  .vector_size = sizeof (u32),
  .format_trace = format_my_ip_trace,
  .type = VLIB_NODE_TYPE_INTERNAL,
  //.type = VLIB_NODE_TYPE_INPUT,
  .n_errors = ARRAY_LEN(my_ip_error_strings),
  .error_strings = my_ip_error_strings,

  .n_next_nodes = MY_N_NEXT,

  /* edit / add dispositions here */
  .next_nodes = {
    //[MY_NEXT_INTERFACE_OUTPUT] = "interface-output",
    [MY_NEXT_INTERFACE_OUTPUT] = "ip4-lookup",
  },
};
```