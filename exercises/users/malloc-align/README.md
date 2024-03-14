
# 16字节对齐

```
    void *mem = malloc(1024+16);
    void *ptr = ((char *)mem+16) & ~(char *)0x0F;
    memset_16aligned(ptr, 0, 1024);
    free(mem);
```

# align1

```
oid* aligned_malloc(size_t required_bytes, size_t alignment) {
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)malloc(required_bytes + offset)) == NULL) {
        return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}
 
void aligned_free(void *p) {
    free(((void**)p)[-1]);
}
```
# aligned-malloc.c

```
[root@centos7 test]# gcc aligned-malloc.c  -o aligned
[root@centos7 test]# ./aligned 
sizeof(int): 4, sizeof(void*) : 8 , sizeof(long): 8 
addr of var1 : 0xffffc06180e4 , 0xffffc06180e5, 0xffffc06180e8 
alined 128: 0 
```

64位机器上指针sizeof(void*)长度：64位  
(char*)&var1 + 1： 先进行强制转换，转换为char*再进行加1操作  
(long)addr & 0x7F： 如果低7位是0，则是128对齐  

# references

[C语言程序开发中，如何对一段内存进行 16 字节对齐操作？](https://blog.popkx.com/c%E8%AF%AD%E8%A8%80%E7%A8%8B%E5%BA%8F%E5%BC%80%E5%8F%91%E4%B8%AD-%E5%A6%82%E4%BD%95%E5%AF%B9%E4%B8%80%E6%AE%B5%E5%86%85%E5%AD%98%E8%BF%9B%E8%A1%8C-16-%E5%AD%97%E8%8A%82%E5%AF%B9%E9%BD%90/)