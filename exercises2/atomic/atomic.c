#include <stdio.h>
#include <stdatomic.h>
#include <stdbool.h>
int fun1(){
    int x = 1;
    int expected = 1;
    int desired = 2;
    bool result = __atomic_compare_exchange_n(&x, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    printf("result=%d, x=%d\n", result, x); // 输出：result=1, x=2
    return 0;
}
int fun2(){
    int x = 1;
    int expected = 3;
    int desired = 2;
    bool result = __atomic_compare_exchange_n(&x, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    printf("result=%d, x=%d\n", result, x); // 输出：result=0, x=1
    return 0;
}
int main() {
    fun1();
    fun2();
    return 0;
}
