#func_name=rollback_expand_heap
func_name=${1}
echo "${func_name}"
sed  -i "s/${func_name}/test_${func_name}/g"  lib/librte_eal//common/eal_common_memzone.c
#sed  -i "s/${func_name}/test_${func_name}/g"  malloc_heap.h
#sed  -i "s/rte_eal_get_configuration/test_rte_eal_get_configuration/g"  eal.c
