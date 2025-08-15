# insmod map_test.ko 
```
[root@centos7 bitmap]# dmesg | tail -n 16
[683642.042893] bit 0 val 1 
[683642.045506] bit 1 val 0 
[683642.048128] bit 2 val 1 
[683642.050736] bit 3 val 0 
[683642.053344] bit 4 val 1 
[683642.055952] bit 5 val 0 
[683642.058567] bit 6 val 1 
[683642.061175] bit 7 val 0 
[683642.063783] bit 0 val 0 
[683642.066395] bit 1 val 0 
[683642.069004] bit 2 val 0 
[683642.071611] bit 3 val 0 
[683642.074219] bit 4 val 0 
[683642.076831] bit 5 val 0 
[683642.079439] bit 6 val 0 
[683642.082047] bit 7 val 0 
```
# insmod map_test2.ko 
```
[root@centos7 bitmap]# dmesg | tail -n 16
[688042.267477] **bit 0 val 1 
[688042.270263] **bit 1 val 0 
[688042.273054] **bit 2 val 1 
[688042.275836] **bit 3 val 0 
[688042.278617] **bit 4 val 1 
[688042.281403] **bit 5 val 0 
[688042.284184] **bit 6 val 1 
[688042.286965] **bit 7 val 0 
[688042.289746] ** bit 0 val 0 
[688042.292619] ** bit 1 val 0 
[688042.295485] ** bit 2 val 0 
[688042.298352] ** bit 3 val 0 
[688042.301224] ** bit 4 val 0 
[688042.304091] ** bit 5 val 0 
[688042.306957] ** bit 6 val 0 
[688042.309823] ** bit 7 val 0 
```

#   BITMAP_FIRST_WORD_MASK
```C
#define BITMAP_FIRST_WORD_MASK(start) (~0UL << ((start) & (BITS_PER_LONG - 1)))
```
****start 参数代表从 bit 0 到 bit (start - 1) 的位都清零****

BITMAP_FIRST_WORD_MASK 宏用于制作一个 BIT_PER_LONG 位掩码，掩码从 bitmap 的低 位清零, 32 位系统上，mask 就是 32 位；64 位系统上，mask 就是 64 位。 start 参数代表从 bit 0 到 bit (start - 1) 的位都清零. BITMAP_FIRST_WORD_MASK(0) 在 32 位系统上获得的掩码就是 0xffffffff; 在 64 位系统上获得的掩码就是 0xffffffffffffffff。函数首先通过将一个 unsigned long 的变量 0 取反，得到 与 BITS_PER_LONG 一样长度的值，其值所有 bit 都为 1，然后通过将该值左移 start 位， 左移之后再与 (BITS_PER_LONG - 1) 相与。

```
static int __init my_init(void)
{
    printk(" BITS_PER_LONG %d\n", BITS_PER_LONG);
    printk("Bitmap(0):   %#lx\n", BITMAP_FIRST_WORD_MASK(0));
    printk("Bitmap(1):   %#lx\n", BITMAP_FIRST_WORD_MASK(1));
    printk("Bitmap(2):   %#lx\n", BITMAP_FIRST_WORD_MASK(2));
    printk("Bitmap(3):   %#lx\n", BITMAP_FIRST_WORD_MASK(3));
    return 0;
}
```

```
[ 2309.517354]  BITS_PER_LONG 64
[ 2309.520309] Bitmap(0):   0xffffffffffffffff
[ 2309.524478] Bitmap(1):   0xfffffffffffffffe
[ 2309.528642] Bitmap(2):   0xfffffffffffffffc
[ 2309.532809] Bitmap(3):   0xfffffffffffffff8
```

#   BITMAP_LAST_WORD_MASK

```
#define BITMAP_LAST_WORD_MASK(nbits) (~0UL >> (-(nbits) & (BITS_PER_LONG - 1)))
```
BITMAP_LAST_WORD_MASK 宏用于获得含 1 的掩码。参数 nbits 代表从右边起，含有 1 的个数。函数首先将 0UL 取反，以此获得长度为 BITS_PER_LONG 的全 1 值，然后 将该值进行右移，右移的位数为 -nibits，也就说明这样做的结果是从 bit0 向左获得 特定个数个 1. BITMAP_LAST_WORD_MASK(1) 的值就是 0x1, BITMAP_LAST_WORD_MASK(2) 的值就是 0x3.


```
        printk("Bitmap(0):   %#lx\n", BITMAP_LAST_WORD_MASK(0));
        printk("Bitmap(1):   %#lx\n", BITMAP_LAST_WORD_MASK(1));
        printk("Bitmap(2):   %#lx\n", BITMAP_LAST_WORD_MASK(2));
        printk("Bitmap(3):   %#lx\n", BITMAP_LAST_WORD_MASK(3));
        printk("Bitmap(4):   %#lx\n", BITMAP_LAST_WORD_MASK(4));
```

```
[ 3486.148520] Bitmap(0):   0xffffffffffffffff
[ 3486.152684] Bitmap(1):   0x1
[ 3486.155561] Bitmap(2):   0x3
[ 3486.158430] Bitmap(3):   0x7
[ 3486.161297] Bitmap(4):   0xf
```

# __bitmap_set

__bitmap_set() 用于置位 bitmap 从 start 开始，长度为 len 的 bit。参数 map 指向 bitmap；参数 start 指向开始置位的位；参数 len 代表需要置位的数量。 函数首先调用 BIT_WORD() 计算 start bit 所在的 long 偏移，然后计算 start 在 long 内剩余的 bit 数。调用 BITMAP_FIRST_WORD_MASK() 获得从低位为 0 的 掩码。函数调用 while 循环，如果 len 超出了 BITS_PER_LONG 的长度，那么就 将超出的部分完整 long 的 bits 都设置为 1. 如果剩余的 bit 不满足 long 长度， 那么就是将 mask_to_set 与 BITMAP_LAST_WORD_MASK 进行相与，最后相或，以此 置位指定的位数。

```
void __bitmap_set(unsigned long *map, unsigned int start, int len)
{
        unsigned long *p = map + BIT_WORD(start);
        const unsigned int size = start + len;
        int bits_to_set = BITS_PER_LONG - (start % BITS_PER_LONG);
        unsigned long mask_to_set = BITMAP_FIRST_WORD_MASK(start);

        while (len - bits_to_set >= 0) {
                *p |= mask_to_set;
                len -= bits_to_set;
                bits_to_set = BITS_PER_LONG;
                mask_to_set = ~0UL;
                p++;
        }
        if (len) {
                mask_to_set &= BITMAP_LAST_WORD_MASK(size);
                *p |= mask_to_set;
        }
}
EXPORT_SYMBOL(__bitmap_set);
```

## test1

```
printk("test begin >>>>>>>>>>>>>>>>>>> \n");
        unsigned long bitmap1 = 0xffff0001;
        unsigned long bitmap2 = bitmap1;
        /* set special bits */
        __bitmap_set(&bitmap1, 4, 4);
        printk("%#lx set 4 bit: %#lx\n", bitmap2, bitmap1);
```


```
[ 3814.021548] 0xffff0001 set 4 bit: 0xffff00f1
```

## test2

```
        unsigned long bits[2];
        printk("test begin >>>>>>>>>>>>>>>>>>> \n");
        memset(bits, 0, sizeof(bits));
        printk("before set %#lx, %#lx\n", bits[0], bits[1]);
        /* set special bits */
        __bitmap_set(bits, 4, 64);
        printk("after set %#lx, %#lx\n", bits[0], bits[1]);
```

```
[ 4189.738374] before set 0x0, 0x0
[ 4189.741502] after set 0xfffffffffffffff0, 0xf
```

#  find_next_bit()

参数：

@addr：位图（数组）的起始地址。
size:位图的大小，即位图中有效bit位的个数。注意，Linux内核实际调用该函数时，该参数的值不一定是32的整数倍（32位系统下）。假设构成位图的数组大小为3，即一共有96个bit，但函数调用时，参数size可           能是90，那么，从逻辑上说，数组最后一个元素的最后6位是不参与构成位图的，即它们不是位图的组成部分，是“无效”的；而前边的90个bit共同构成了位图，它们是“有效”的。注意，后面解释中经常会用 到“有效位”和“无效位”的概念，对此，读者一定要理解清楚。

offset:查找起点。即从位图中索引为offset的位（包括该位）开始，查找第一个为1的bit位，offset之前的bit位不在搜索范围之内。“查找起点”这个概念在后面的叙述中经常会用到，希望读者能理解清楚。

返回值：找到的bit位的索引。


```
unsigned long find_next_bit(const unsigned long *addr, unsigned long size,
        unsigned long offset)
{

    /*

   * 1.addr数组中的每一个元素为一个bitmap,可以表示sizeof(unsigned long)*8个元素

   * 2.size为最大尺寸,可以是[1,数组元素个数*sizeof(unsigned long)*8]中的任意值

   * 3.offset=[0,size]

   */

const unsigned long *p = addr +BITOP_WORD(offset);//定位到起始元素

unsigned long result = offset &~(BITS_PER_LONG-1);//已经跳过的bit位个数

unsigned long tmp;

 

if (offset >= size)//起始偏移超过了最大尺寸,直接返回最大尺寸

           return size;

```

```

        unsigned long bitmap1 = 0xffff00f1;
        unsigned int pos;

        printk("test begin >>>>>>>>>>>>>>>>>>> \n");
        /* Find first bit position. */
        pos = find_next_bit(&bitmap1, 32, 0);
        printk("Find %#lx first bit postion is: %d\n", bitmap1, pos);
        pos = find_next_bit(&bitmap1, 32, 12);
        printk("Find %#lx first bit postion is: %d\n", bitmap1, pos);
        pos = find_next_bit(&bitmap1, 32, 8);
        printk("Find %#lx first bit postion is: %d\n", bitmap1, pos);
```


```
[ 7590.531741] test begin >>>>>>>>>>>>>>>>>>> 
[ 7590.535920] Find 0xffff00f1 first bit postion is: 0
[ 7590.540776] Find 0xffff00f1 first bit postion is: 16
[ 7590.545726] Find 0xffff00f1 first bit postion is: 16
```