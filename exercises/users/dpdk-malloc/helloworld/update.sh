#func_name=rollback_expand_heap
func_name=${1}
echo "${func_name}"
sed  -i "s/${func_name}/test_${func_name}/g"  malloc_heap.c
sed  -i "s/${func_name}/test_${func_name}/g"  malloc_heap.h
