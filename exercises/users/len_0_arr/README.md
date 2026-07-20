
# run

```
[root@centos7 test]# gcc len_0_arr.c  -o len_0_arr
[root@centos7 test]# ./len_0_arr 
the length of struct test1:4
the length of struct test2:16
the length of struct test3:16
var1->b[0]=0    var1->b[1]=1    var1->b[2]=2    var1->b[3]=3    var1->b[4]=4    var1->b[5]=5    var1->b[6]=6    var1->b[7]=7    var1->b[8]=8    var1->b[9]=9
p var1 = 0x16490010
p var1->a 0x16490010
var1->b 0x16490014
p var1->b 0x16490014
p var1->b[0] 0x16490014
p var1->b[1] 0x16490018


var2->b[0]=0    var2->b[1]=1    var2->b[2]=2    var2->b[3]=3    var2->b[4]=4    var2->b[5]=5    var2->b[6]=6    var2->b[7]=7    var2->b[8]=8    var2->b[9]=9
p var2 = 0x16490050
p var2->a 0x16490050
var2->b 0x16490070
p var2->b 0x16490058
p var2->b[0] 0x16490070
p var2->b[1] 0x16490074
[root@centos7 test]# 
```


```C
struct test1
{
    int a;
    int b[0];
};

the length of struct test1:4
the length of struct test2:16
the length of struct test3:16
var1->b[0]=0    var1->b[1]=1    var1->b[2]=2    var1->b[3]=3    var1->b[4]=4    var1->b[5]=5    var1->b[6]=6    var1->b[7]=7    var1->b[8]=8    var1->b[9]=9
p var1 = 0x16490010
p var1->a 0x16490010
var1->b 0x16490014
p var1->b 0x16490014
p var1->b[0] 0x16490014
p var1->b[1] 0x16490018
```

***var1->b  和 var1->b[0]指向同一个位置***


1） 使用长度为0的数组可以比指针更方便地进行内存的管理。  
*a*结构体test1在分配内存时，则是采用一次分配的原则，一次性将所需的内存全部分配给它，释放也是一次释放。数组和结构体的内存是连续的。
*b*
结构体test2在分配内存时，需采用两步：首先，需为结构体分配一块内存空间；其次再为结构体中的成员变量分配内存空间。这样两次分配的内存是不连续的，需要分别对其进行管理。当使用长度为0的数组时，则是采用一次分配的原则，一次性将所需的内存全部分配给它。相反，释放时也是一样的。 
2) 长度为0的数组并不占有内存空间，而指针方式需要占用内存空间 