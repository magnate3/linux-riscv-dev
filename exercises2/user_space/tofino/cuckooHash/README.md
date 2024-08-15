
# CuckooHash

[CuckooHash](https://blog.csdn.net/Jacksqh/article/details/130616495)    

## 扩容
假如我们在一次插入的时候kick out元素的次数到达一定的阈值（比如是501次），但是我们在初始的时候设置的kick out 的最大次数是500次，那么这时候就需要将hash桶给进行扩容（说明冲突太多了需要调整好桶的大小），再将桶中的数据进行重新映射插入


## tofino  CuckooHash   


[Switcharoo-P4](https://github.com/Switcharoo-P4/Switcharoo-P4)   