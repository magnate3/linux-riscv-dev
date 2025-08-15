### 32位和64位的区别

位置：nginx/src/core/nginx_config.h
```C
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;
```

```C
intptr_t和uintptr_t在linux平台的/usr/include/stdint.h头文件中可以找到

/* Types for `void *' pointers.  */
#if __WORDSIZE == 64
# ifndef __intptr_t_defined
typedef long int		intptr_t;
#  define __intptr_t_defined
# endif
typedef unsigned long int	uintptr_t;
#else
# ifndef __intptr_t_defined
typedef int			intptr_t;
#  define __intptr_t_defined
# endif
typedef unsigned int		uintptr_t;
#endif
```

```C
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))
```

其中 

- `uintptr_t`  是 `unsigned int` 的 `typedef` 
- `u_char *` 是 `char *` 的 `typedef`

所以宏之中发生了 `int -> char *` 的强转：

- `32` 位环境中是不存在问题的，因为都为 `4` 字节
- `64` 位环境中存在问题：由于指针 `8` 字节，而 `int` 为 `4` 字节，所以引发精度丢失

### 64位系统 

`x64` 环境中直接将 `uintptr_t` 直接定义为 `unsigned long int`

```C
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((unsigned long int) (p) + ((unsigned long int) a - 1)) & ~((unsigned long int) a - 1))
```
