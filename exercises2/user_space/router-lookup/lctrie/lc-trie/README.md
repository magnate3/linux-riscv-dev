1. 功能说明
1.1 为了方便实验和更好的理解trie, 本模块从路由LC-trie协议,去除了路由规则,trie的扩展/压缩等复杂模块简化而成
1.2 主要完成一个32bits的字典树(网络搜索参考字典树的逻辑及下文的"实例图解")
1.3 本模块功能包括:插入节点,移除节点和节点查找,节点可以附带具体业务对象数据
1.4 本模块包含基础插入/移除和查找api,以及一个使用实例
2. 调试环境信息:ARM64,linux 5.0.0
3. 运行输出信息:为了便于理解, 本模块在init函数中执行实例程序,并输出人性化的调试信息,如下[运行输出信息]
4. 代码说明
bits32_trie.c: 模块初始化代码及测试实例
bits32_trie_new.c:节点的创建,内存分配,本例分配内存统一使用kmalloc/kfree,如需优化,自行修改
bits32_trie_debug.c:调试信息输出,针对trie结构体进行逐层打印
bits32_trie_insert.c:节点插入代码
bits32_trie_lookup.c:节点查找代码
bits32_trie_remove.c:节点删除代码

运行输出信息
```
/lib/modules/5.0.0/extra # insmod BiscuitOS-modules-0.0.1.ko 
[   25.401183] BiscuitOS_modules_0.0.1: loading out-of-tree module taints kernel.
[   25.424002] bits32_trie_init:[16 - 4]
[   25.424026] ~~~~~~~~  start insert  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[   25.424432] bits32_trie_init insert leaf[0x12345678] data[0x0]
[   25.424979] bits32_trie_init insert leaf[0x87654321] data[0x1]
[   25.425388] bits32_trie_init insert leaf[0x11111111] data[0x2]
[   25.425647] bits32_trie_init insert leaf[0x22222222] data[0x3]
[   25.425854] ~~~~~~~~   end insert   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[   25.426055] --------------------------------------------------------------
[   25.426273] ~~~~~~~~   now let's look the trie view ~~~~~~~~~~~~~~~~~
[   25.426754] tnode[0x0]:pos[0x1f]bits[0x1]cindex[0x0]
[   25.426882] - tnode[0x0]:pos[0x1d]bits[0x1]cindex[0x0]
[   25.427160] - - tnode[0x10000000]:pos[0x19]bits[0x1]cindex[0x0]
[   25.427519] - - - leaf[0x11111111]:pos[0x0]bits[0x0]data[0x2]cindex[0x0]
[   25.427921] - - - leaf[0x12345678]:pos[0x0]bits[0x0]data[0x0]cindex[0x1]
[   25.428277] - - leaf[0x22222222]:pos[0x0]bits[0x0]data[0x3]cindex[0x1]
[   25.428625] - leaf[0x87654321]:pos[0x0]bits[0x0]data[0x1]cindex[0x1]
[   25.428948] ~~~~~~~~   trie view end, it's as what you think??  ~~~~~
[   25.429585] --------------------------------------------------------------
[   25.429815] ~~~~~~~~   test lookup  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[   25.430032] bits32_trie_init lookup[0x11111111]index[0x2]
[   25.430340] bits32_trie_init it seems ok, found leaf->key[0x11111111]
[   25.430570] ~~~~~~~~  test remove  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[   25.430782] bits32_trie_init remove[0x11111111]index[0x2]
[   25.431139] bits32_trie_init it seems ok, found no leaf->key[0x11111111] after del
[   25.431379] bits32_trie_init remove[12345678]index[0]
[   25.431649] bits32_trie_init it seems ok, found no leaf->key[0x12345678] after del
[   25.431886] ~~~~~~~~  now let's check the trie if it same as what you think  ~~~~
[   25.432127] tnode[0x0]:pos[0x1f]bits[0x1]cindex[0x0]
[   25.432139] - tnode[0x0]:pos[0x1d]bits[0x1]cindex[0x0]
[   25.432308] - - leaf[0x22222222]:pos[0x0]bits[0x0]data[0x3]cindex[0x1]
[   25.432474] - leaf[0x87654321]:pos[0x0]bits[0x0]data[0x1]cindex[0x1]
[   25.432704] bits32_trie_init finish
/lib/modules/5.0.0/extra # 
```

实例图解
             bit31  ->                      bit0
0x12345678   00010010 00110100 01010110 01111000
0x87654321   10000111 01100101 01000011 00100001
0x11111111   00010001 00010001 00010001 00010001
0x22222222   00100010 00100010 00100010 00100010


	               tp
	               |
	               v
              0x12345678
(00010010 00110100 01010110 01111000)

                                           tp
                                           |
                                           v
	               ------------------ 0x0 --------------------
	               |                                         |
	               v                                         v
              0x12345678                                0x87654321
(00010010 00110100 01010110 01111000)    (10000111 01100101 01000011 00100001)


                                                               tp
                                                               |
                                                               v
	                                   ------------------ 0x0 --------------------
	                                   |                                         |
	                                   v                                         v
                                  0x10000000                                0x87654321
                    (00010000 00000000 00000000 00000000)    (10000111 01100101 01000011 00100001)
	                                   |
	                                   v
	               -------------------- ----------------------
	               |                                         |
	               v                                         v
              0x12345678                                0x11111111
(00010010 00110100 01010110 01111000)    (00010001 00010001 00010001 00010001)


                                                                                    tp
                                                                                    |
                                                                                    v
	                                                        ------------------ 0x0 --------------------
	                                                        |                                         |
	                                                        v                                         v
                                                               0x0                                   0x87654321
                                                  (00000000 00000000 00000000 00000000)       (10000111 01100101 01000011 00100001)
                                                                |
                                                                v
                                           --------------------  ---------------------
	                                   |                                         |
	                                   v                                         v
                                     0x10000000                                 0x22222222
                    (00000001 00000000 00000000 00000000)    (00100010 00100010 00100010 00100010)
                                           |
                                           v
	               -------------------- ----------------------
	               |                                         |
	               v                                         v
                   0x11111111                                0x12345678
     (00010001 00010001 00010001 00010001)    (00010010 00110100 01010110 01111000)
	 
	 
# 体系结构

```
static inline unsigned long __fls(unsigned long word)
{
    int num = BITS_PER_LONG - 1;

#if defined(X86_64) || defined(ARM64)
    if (!(word & (~0ul << 32))) {
        num -= 32;
        word <<= 32;
    }
#endif
    if (!(word & (~0ul << (BITS_PER_LONG-16)))) {
        num -= 16;
        word <<= 16;
    }
    if (!(word & (~0ul << (BITS_PER_LONG-8)))) {
        num -= 8;
        word <<= 8;
    }
    if (!(word & (~0ul << (BITS_PER_LONG-4)))) {
        num -= 4;
        word <<= 4;
    }
    if (!(word & (~0ul << (BITS_PER_LONG-2)))) {
        num -= 2;
        word <<= 2;
    }
    if (!(word & (~0ul << (BITS_PER_LONG-1))))
        num -= 1;
    return num;
}
```

## 有-DARM64
```
[root@centos7 lc-trie]# ./bits32_trie 
main:[16 - 4]~~~~~~~~  start insert  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main insert leaf[0x12345678] data[0x0]
main insert leaf[0x87654321] data[0x1]
main insert leaf[0x11111111] data[0x2]
main insert leaf[0x22222222] data[0x3]
~~~~~~~~   end insert   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--------------------------------------------------------------
~~~~~~~~   now let's look the trie view ~~~~~~~~~~~~~~~~~
tnode[0x0]:pos[0x1f]bits[0x1]cindex[0x0]- tnode[0x0]:pos[0x1d]bits[0x1]cindex[0x0]- - tnode[0x10000000]:pos[0x19]bits[0x1]cindex[0x0]- - - leaf[0x11111111]:pos[0x0]bits[0x0]data[0x2]cindex[0x0]- - - leaf[0x12345678]:pos[0x0]bits[0x0]data[0x0]cindex[0x1]- - leaf[0x22222222]:pos[0x0]bits[0x0]data[0x3]cindex[0x1]- leaf[0x87654321]:pos[0x0]bits[0x0]data[0x1]cindex[0x1]~~~~~~~~   trie view end, it's as what you think??  ~~~~~
--------------------------------------------------------------
~~~~~~~~   test lookup  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main lookup[0x11111111]index[0x2]
main it seems ok, found leaf->key[0x11111111]
~~~~~~~~  test remove  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main remove[0x11111111]index[0x2]
main it seems ok, found no leaf->key[0x11111111] after del
main remove[12345678]index[0]
main it seems ok, found no leaf->key[0x12345678] after del
~~~~~~~~  now let's check the trie if it same as what you think  ~~~~
tnode[0x0]:pos[0x1f]bits[0x1]cindex[0x0]- tnode[0x0]:pos[0x1d]bits[0x1]cindex[0x0]- - leaf[0x22222222]:pos[0x0]bits[0x0]data[0x3]cindex[0x1]- leaf[0x87654321]:pos[0x0]bits[0x0]data[0x1]cindex[0x1]main finish
```

## 没有-DARM64


![images](test1.png)






