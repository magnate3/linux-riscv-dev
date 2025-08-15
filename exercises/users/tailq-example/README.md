#  TAILQ_ENTRY


```
#include<stdio.h>
#include <sched.h>
#include <sys/queue.h>
struct student{
    TAILQ_ENTRY(student) link;
};
int main()
{
    return 0;
}
```
```
[root@centos7 ~]# gcc -E test2.c -o  test2.i 
[root@centos7 ~]# cat test2.i | tail -n 10
# 3 "test2.c" 2
# 1 "/usr/include/sys/queue.h" 1 3 4
# 4 "test2.c" 2
struct student{
    struct { struct student *tqe_next; struct student * *tqe_prev; } link;
};
int main()
{
    return 0;
}
```

