
# list_for_each_entry pk list_for_each_entry_safe
相比于list_for_each_entry，list_for_each_entry_safe用指针n对链表的下一个数据结构进行了临时存储，所以如果在遍历链表的时候需要做删除链表中的当前项操作时，用list_for_each_entry_safe可以安全的删除，而不会影响接下来的遍历过程（用n指针可以继续完成接下来的遍历， 而list_for_each_entry则无法继续遍历，删除后会导致无法继续遍历）。
 

# insmod  list_test.ko 

```
[root@centos7 list]# insmod  list_test.ko 
[root@centos7 list]# rmmod  list_test.ko 
[root@centos7 list]# dmesg | tail -n 10
[257983.920984] Loading Module
[257983.923768] Day: 2 Month: 10 Year: 1988 Gender: male Name: Tom 
[257983.929761] Day: 5 Month: 6 Year: 1999 Gender: female Name: kim 
[257983.935829] Day: 20 Month: 2 Year: 1995 Gender: male Name: Adam 
[257983.941900] Day: 12 Month: 1 Year: 1971 Gender: female Name: Laura 
[257987.589486] Removing Module
[257987.592355] delete node Day: 2 Month: 10 Year: 1988 Gender: male Name: Tom 
[257987.599383] delete node Day: 5 Month: 6 Year: 1999 Gender: female Name: kim 
[257987.606486] delete node Day: 20 Month: 2 Year: 1995 Gender: male Name: Adam 
[257987.613593] delete node Day: 12 Month: 1 Year: 1971 Gender: female Name: Laura 
[root@centos7 list]# 
```

#  insmod  list_test2.ko

```
[root@centos7 list]# dmesg | tail -n 45
[258248.950446] [list_dev_init:154]: list init[E]
[258248.950447] [list_test:46]: Test List [E]
[258248.958955] [list_test:53]: create list
[258248.962876] [list_test:67]: traversal list
[258248.967040] [list_test:71]: Dump List n1 used list_for_each
[258248.972677] [list_test:75]: n1 is 0
[258248.976235] [list_test:75]: n1 is 1
[258248.979794] [list_test:75]: n1 is 2
[258248.983357] [list_test:75]: n1 is 3
[258248.986917] [list_test:75]: n1 is 4
[258248.990474] [list_test:79]: Dump List n2 used list_for_each_entry
[258248.996630] [list_test:81]: n2 is 5
[258249.000188] [list_test:81]: n2 is 6
[258249.003753] [list_test:81]: n2 is 7
[258249.007313] [list_test:81]: n2 is 8
[258249.010870] [list_test:81]: n2 is 9
[258249.014434] [list_test:105]: splice list
[258249.018425] [list_test:110]: Dump List n1 used list_for_each_entry
[258249.024667] [list_test:115]: Dump List n2 used list_for_each_entry
[258249.030904] [list_test:117]: n2 is 0 ////////////////////
[258249.034554] [list_test:117]: n2 is 1
[258249.038200] [list_test:117]: n2 is 2
[258249.041846] [list_test:117]: n2 is 3
[258249.045496] [list_test:117]: n2 is 4
[258249.049143] [list_test:117]: n2 is 5
[258249.052792] [list_test:117]: n2 is 6
[258249.056438] [list_test:117]: n2 is 7
[258249.060084] [list_test:117]: n2 is 8
[258249.063734] [list_test:117]: n2 is 9
[258249.067381] [list_test:122]: rotate_left list
[258249.071803] [list_test:126]: Dump List n2 used list_for_each_entry
[258249.078044] [list_test:128]: n2 is 1
[258249.081690] [list_test:128]: n2 is 2
[258249.085340] [list_test:128]: n2 is 3
[258249.088987] [list_test:128]: n2 is 4
[258249.092637] [list_test:128]: n2 is 5
[258249.096283] [list_test:128]: n2 is 6
[258249.099928] [list_test:128]: n2 is 7
[258249.103579] [list_test:128]: n2 is 8
[258249.107225] [list_test:128]: n2 is 9
[258249.110871] [list_test:128]: n2 is 0 ////////////////////
[258249.114522] [list_test:133]: delete list
[258249.118513] [list_test:137]: Delete List n1 used list_for_each_entry_safe
[258249.125360] [list_test:142]: Delete List n2 used list_for_each_entry_safe
[258249.132203] [list_test:149]: Test List [X]
```