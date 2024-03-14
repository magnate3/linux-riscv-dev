
# insmod  sparse_test.ko 
 
```
/* bitwise: big-endian, little-endian */
typedef unsigned int __bitwise bs_t;
static int __init my_init(void)
{
        bs_t a = (__force bs_t)0x12345678;
        bs_t b;

#ifdef __LITTLE_ENDIAN
        printk("little-endian original: %#x\n", a);
#else
        printk("big-endian original:    %#x\n", a);
#endif
        /* Cover to little-endian */
        b = (__force bs_t)cpu_to_le32(a);
        printk("%#x to little-endian: %#x\n", a, b);
        /* Cover to big-endian */
        b = (__force uint32_t)cpu_to_be32(a);
        printk("%#x to bit-endian:    %#x\n", a, b);

        return 0;
}
```


```
[600610.251430] sparse_test: no symbol version for module_layout
[600610.257165] sparse_test: loading out-of-tree module taints kernel.
[600610.263452] sparse_test: module verification failed: signature and/or required key missing - tainting kernel
[600610.273803] little-endian original: 0x12345678
[600610.278320] 0x12345678 to little-endian: 0x12345678
[600610.283264] 0x12345678 to bit-endian:    0x78563412
```


```
        if (macb_is_gem(bp))
        {
                if (native_io) {
                //bp->macb_reg_readl = hw_readl_native;
                //bp->macb_reg_writel = hw_writel_native;
                pr_info("macb is gem and native io, base addr %p and vlaue : %x \n", mem,  __raw_readl(mem + MACB_NCR));

                } else {
                //bp->macb_reg_readl = hw_readl;
                //bp->macb_reg_writel = hw_writel;
                pr_info("macb is gem and not  native io ,base addr %p and vlaue : %x \n", mem,  readl_relaxed(mem + MACB_NCR));
                }
        }
        else
        {
                if (native_io) {
                //bp->macb_reg_readl = hw_readl_native;
                //bp->macb_reg_writel = hw_writel_native;
                pr_info("macb is not  gem and native io, base addr %p and vlaue : %x \n", mem,  __raw_readl(mem + GEM_NCR));

                } else {
                //bp->macb_reg_readl = hw_readl;
                //bp->macb_reg_writel = hw_writel;
                pr_info("macb is not gem and not  native io ,base addr %p and vlaue : %x \n", mem,  readl_relaxed(mem + GEM_NCR));
                }
        }
```

进行Linux驱动方面开发的程序员，经常需要使用readl/writel系列函数对Memory-Mapped IO进行读写：

1) raw前缀只与byteorder相关，即readl/writel是linux默认的小端操作，而raw_readl/raw_writel是native访问。也就是说：如果是小端系统，raw_readl与readl相同，如果是大端系统，raw_readl与readl的有字节序差别。

2) 双下划线前缀与指令保序相关，即readl/writel包含存储器栅栏指令mb，能够保证IO读写顺序，而__readl/__writel则不能保证。

另外，有的体系结构还会定义readl_relaxed/writel_relaxed接口，其含义应该与__readl/__writel相同，表示小端、不带存储器栅栏的读写。