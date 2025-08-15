## 简介

源代码分布在多级目录下时

Makefile文件的编写方式。

这里是源代码，主要分2大类来写：

 * 集中式–单个Makefile统一管理
 * 分布式–多个Makefile分散管理
 
## mk-centralization 
 
###  mkdir bin

```
[root@centos7 mk-centralization]# gcc -Wall -I./include -g -c bar/bar.c -o bin/bar.o 
Assembler messages:
Fatal error: can't create bin/bar.o: No such file or directory
[root@centos7 mk-centralization]# ls
bar  foo  include  main.c  Makefile
[root@centos7 mk-centralization]# mkdir bin
[root@centos7 mk-centralization]# gcc -Wall -I./include -g -c bar/bar.c -o bin/bar.o 
[root@centos7 mk-centralization]# 
[root@centos7 mk-centralization]# mkdir bin
[root@centos7 mk-centralization]# gcc -Wall -I./include -g -c bar/bar.c -o bin/bar.o 
[root@centos7 mk-centralization]# make
gcc -Wall -I./include -g -c bar/commonbar.c -o bin/commonbar.o 
gcc -Wall -I./include -g -c foo/commonfoo.c -o bin/commonfoo.o 
gcc -Wall -I./include -g -c foo/foo.c -o bin/foo.o 
gcc -Wall -I./include -g -c main.c -o main.o 
gcc bin/bar.o bin/commonbar.o bin/commonfoo.o bin/foo.o main.o -o bin/main
```

## 博客文章

[多级目录Makefile编写方法](http://xnzaa.github.io/2015/01/26/Makefile%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/)
