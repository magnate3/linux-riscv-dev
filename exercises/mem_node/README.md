# insmod  print_node_test.ko

```
[root@centos7 mem_node]# insmod  print_node_test.ko
[root@centos7 mem_node]# dmesg | tail -n 20
[79496.523943] address = ffff803fffffe500
[79496.527676] node id = 0
[79496.530112] nr_zones = 2
[79496.532632] nod present pages = 2097085
[79496.536450] nod spanned pages = 4194304
[79496.540268] address = ffff805fffffe500
[79496.544006] node id = 1
[79496.546444] nr_zones = 2
[79496.548965] nod present pages = 2097152
[79496.552789] nod spanned pages = 2097152
[79496.556608] address = ffffa03fffffe500
[79496.560341] node id = 2
[79496.562783] nr_zones = 2
[79496.565304] nod present pages = 2097152
[79496.569122] nod spanned pages = 2097152
[79496.572944] address = ffffa05fffffe500
[79496.576676] node id = 3
[79496.579112] nr_zones = 2
[79496.581638] nod present pages = 2097152
[79496.585456] nod spanned pages = 2097152
[root@centos7 mem_node]#
```