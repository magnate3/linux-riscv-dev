
```
static constexpr Timestamp MAX_DURATION = 0x7000000000000000;

static bool TimestampGT(Timestamp t_0, Timestamp t_1) {
  return t_0 - t_1 < MAX_DURATION && t_0 != t_1;
}
```
说明：Timestamp类型为std::uint64_t  即无符号64位整型，t_0与t_1为两个测量时间戳，差值不会很大。
概念：系统时间timestamp为模块处理完的时间，测量时间measured_timestamp 为传感器的观测时间，
他们的数据类型均为Timestamp

在计算机内部，无符号数和其它类型一样，统一都是按照补码加减法规则计算的（毕竟所有的数都是按照补码存放的）而唯一不同的是，最后的结果的解释方式不同。0x7000000000000000代表16进制数，其中0x为前缀（或者用H作为后缀表示16进制数），它转化为二进制数正好是64位数（4×16=64）。
1、当t_0 - t_1 < 0
t_0 - t_1 的二进制数补码最高位必为1（当解释为有符号数时，这个1仅代表负数的意思），即t_0 - t_1 的值在计算机中存储为1_ _ _ … _ ，而0x7000000000000000作为无符号数在计算机中存储为0111 _ _ _ … _，此时上面的t_0 - t_1 < MAX_DURATION会为false

2、当t_0 - t_1 > 0
t_0 - t_1 的二进制补码最高位为0，即t_0 - t_1 的值在计算机中的存储为0_ _ _ … _ ，而0x7000000000000000作为无符号数在计算机中存储为0111 _ _ _ … _，此时上面的t_0 - t_1 < MAX_DURATION可能会为true（t_0 - t_1二进制值补码的第二位为0时），也可能为false（t_0 - t_1二进制值的第二位至第五位均为1时，但此时两个测量时间不会相差这么大，实际中不会出现这种情况）

#  Linux内核如何来防止jiffies溢出

```
#define time_after(a,b)        \
    (typecheck(unsigned long, a) && \
     typecheck(unsigned long, b) && \
     ((long)(b) - (long)(a) < 0))
#define time_before(a,b)    time_after(b,a)

#define time_after_eq(a,b)    \
    (typecheck(unsigned long, a) && \
     typecheck(unsigned long, b) && \
     ((long)(a) - (long)(b) >= 0))
#define time_before_eq(a,b)    time_after_eq(b,a)

#define time_in_range(a,b,c) \
    (time_after_eq(a,b) && \
     time_before_eq(a,c))

/* Same as above, but does so with platform independent 64bit types.
 * These must be used when utilizing jiffies_64 (i.e. return value of
 * get_jiffies_64() */
#define time_after64(a,b)    \
    (typecheck(__u64, a) &&    \
     typecheck(__u64, b) && \
     ((__s64)(b) - (__s64)(a) < 0))
#define time_before64(a,b)    time_after64(b,a)

#define time_after_eq64(a,b)    \
    (typecheck(__u64, a) && \
     typecheck(__u64, b) && \
     ((__s64)(a) - (__s64)(b) >= 0))
#define time_before_eq64(a,b)    time_after_eq64(b,a)
```

  
对于time_after等比较jiffies先/后的宏，两个值的取值应当满足以下限定条件：  
1) 两个值之间相差从逻辑值来讲应小于有符号整型的最大值。  
2) 对于32位无符号整型，两个值之间相差从逻辑值来讲应小于2147483647。  
对于HZ=100，那么两个时间值之间相差不应当超过2147483647/100秒 = 0.69年 = 248.5天。对于HZ=60，那么两个时间值之间相差不应当超过2147483647/60秒 =   1.135年。在实际代码应用中，需要比较先/后的两个时间值之间一般都相差很小，范围大致在1秒~1天左右，所以以上time_after等比较时间先后的宏完全可以放心地用于实际的代码中。  
 
 
 