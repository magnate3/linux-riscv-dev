

# 宏  

```
#define _POSIX_C_SOURCE 199309L
```

如果是使用cmake，可以在CMakeLists.txt里加上这句    
add_compile_options(-D_POSIX_C_SOURCE=199309L) 