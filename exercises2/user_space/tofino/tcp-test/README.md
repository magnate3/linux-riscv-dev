
```
 @pa_no_overlay("egress","meta.cache_length")
 @pa_solitary("egress","meta.cache_length")
@pa_container_size("egress","meta.cache_length",32)
 @pa_no_overlay("egress","meta.sum_cache_len")
 @pa_solitary("egress","meta.sum_cache_len")
@pa_container_size("egress","meta.sum_cache_len",32)
 @pa_no_overlay("egress","meta.read_index")
 @pa_solitary("egress","meta.read_index")
@pa_container_size("egress","meta.read_index",16)
 @pa_no_overlay("egress","meta.update_index")
 @pa_solitary("egress","meta.update_index")
@pa_container_size("egress","meta.update_index",16)
```