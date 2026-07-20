
# riscv64-linux-gnu-gcc -pthread  u_uacce_pin_page.c  -o test_pin

```
root@ubuntu:~/page_pin/uacce_pin_page_test# riscv64-linux-gnu-gcc -pthread  u_uacce_pin_page.c  -o test_pin
u_uacce_pin_page.c: In function ‘main’:
u_uacce_pin_page.c:81:19: warning: format ‘%lx’ expects argument of type ‘long unsigned int’, but argument 2 has type ‘char *’ [-Wformat=]
  printf("u pin: %lx %lx\n", p, size);
                 ~~^         ~
                 %s
root@ubuntu:~/page_pin/uacce_pin_page_test# pwd
/root/page_pin/uacce_pin_page_test
```


```
root@ubuntu:~/page_pin/uacce_pin_page_test# riscv64-linux-gnu-gcc -lpthread  u_uacce_pin_page.c  -o test_pin
u_uacce_pin_page.c: In function ‘main’:
u_uacce_pin_page.c:80:19: warning: format ‘%lx’ expects argument of type ‘long unsigned int’, but argument 2 has type ‘char *’ [-Wformat=]
  printf("u pin: %lx %lx\n", p, size);
                 ~~^         ~
                 %s
/tmp/ccS2lul9.o: In function `.L3':
u_uacce_pin_page.c:(.text+0xa4): undefined reference to `pthread_setaffinity_np'
/tmp/ccS2lul9.o: In function `.L0 ':
u_uacce_pin_page.c:(.text+0x158): undefined reference to `pthread_create'
collect2: error: ld returned 1 exit status
```
***-pthread***

```
root@ubuntu:~/page_pin/uacce_pin_page_test# riscv64-linux-gnu-gcc -pthread  u_uacce_pin_page.c  -o test_pin
u_uacce_pin_page.c: In function ‘main’:
u_uacce_pin_page.c:80:19: warning: format ‘%lx’ expects argument of type ‘long unsigned int’, but argument 2 has type ‘char *’ [-Wformat=]
  printf("u pin: %lx %lx\n", p, size);
                 ~~^         ~
                 %s
root@ubuntu:~/page_pin/uacce_pin_page_test#
```

## ./test_pin

```
root@Ubuntu-riscv64:~/page_pin# grep page_pin_test /proc/devices | awk '{print $1;}'
244
root@Ubuntu-riscv64:~/page_pin# insmod page_pin_test.ko 
root@Ubuntu-riscv64:~/page_pin# grep page_pin_test /proc/devices | awk '{print $1;}'
243
244
root@Ubuntu-riscv64:~/page_pin# mknod --mode=666 /dev/page_pin_test c  243 0
root@Ubuntu-riscv64:~/page_pin# ls /dev/page_pin_test 
/dev/page_pin_test
root@Ubuntu-riscv64:~/page_pin# chmod  +x test_pin 
root@Ubuntu-riscv64:~/page_pin# mknod --mode=666 /dev/page_pin_test c 238 0
root@Ubuntu-riscv64:~/page_pin# ./test_pin
write data is thread 1839
u pin: 3fb01ca000 64000
min page fault before pin: 209
min page fault after pin: 209
```


#  pin_user_pages

#  try_get_page(page)