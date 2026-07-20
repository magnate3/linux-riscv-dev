
# 两次migrate

```
root@ubuntux86:# insmod  test_hmm.ko 
root@ubuntux86:# ./user 
***** before migrate: 
 Physical address is 5876006912
content is hello world 
**** after first migrate: 
 Physical address is 8796093014016
content is 1109migrate hello world 
**** after send migrate: 
 Physical address is 8796093014016
content is -552migrate hello world 
run over 
```
+ 第一次migrate后的地址是8796093014016，第二次migrate后的地址是8796093014016，相同  
+ 第一次migrate后的内容是1109migrate hello world，第二次migrate后的内容是-52migrate hello world ，不相同 
+ 为什么after first migrate 和after send migrate的Physical address is 8796093014016   

# dmesg
```
[ 4693.347570] HMM test module loaded. This is only for testing HMM.
[ 4695.430786]  enter cmp page<1th> begin: 
[ 4695.430795] g_addr 140627131138048 , page start 18446625070336753664, page end 18446625070336757760
[ 4695.430799] buf is hello world 
[ 4695.430827] dmirror migrate return val: 0 
[ 4695.430888]  enter cmp page<2th> begin: 
[ 4695.430891] g_addr 140627131138048 , page start 18446625070336753664, page end 18446625070336757760
[ 4695.430894] buf is 1109migrate hello world 
[ 4695.430908] dmirror migrate return val: 0
```
+ 1 第一次进入：buf is hello world，page start 18446625070336753664, page end 18446625070336757760    
+ 2 第二次进入：buf is 1109migrate hello world，page start 18446625070336753664, page end 18446625070336757760
+ 3 两次进入的page start 18446625070336753664和 page end 18446625070336757760是一样的     

 