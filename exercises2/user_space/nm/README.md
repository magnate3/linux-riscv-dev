

只查看导出的动态符号 (推荐):   
```
root@ubuntux86:# nm -D  lib64/libcudart.so 
                 w __cxa_finalize
                 w __gmon_start__
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
root@ubuntux86:# 
```