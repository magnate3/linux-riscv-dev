
```
[165155.409062] 1664099075 819299:create task 1 success
[165155.414092] 1664099075 824330:create task 2 success
[165155.419041] task1 enter rcu 
[165156.419075] task2 enter rcu 
[165165.422327] ptest is null 
[165165.425114] ptest2 is ffffa05fc4974200 
[165175.429347] after rcu unlock,  ptest is null 
[165175.492676] rcu_demo_free 1664099095 903257:free ptest
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/rcu_test2/test2.png)