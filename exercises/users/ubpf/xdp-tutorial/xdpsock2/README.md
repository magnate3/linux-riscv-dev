
# #define BATCH_SIZE 1
```
root@ubuntux86:# ./xdpsock_user  -i lo  -q 0  -S
sec:xdp
finding qidconf_map 
lo,1
xsk_configure socket fd 8 
 xsk_configure socket fd 9 
 Writing = 0xe63d3000
length = 42
addr=2097152 | 01 01 01 01 01 01 04 01 03 02 12 5D 08 06 00 01 08 00 06 04 00 01 36 15 FD 2A EE A3 C0 A8 08 64  | ...........]..........6.

                                                                                                                                           addr=2097152 | 00 00 00 00 00 00 D8 3A D4 64 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __  | ......


idx = 1025
```
没有出现read  

# #define BATCH_SIZE 2
```
root@ubuntux86:# ./xdpsock_user  -i lo  -q 0  -S
sec:xdp
finding qidconf_map 
lo,1
xsk_configure socket fd 8 
 xsk_configure socket fd 9 
 Writing = 0x915ff000
length = 42
addr=2097152 | 01 01 01 01 01 01 04 01 03 02 12 5D 08 06 00 01 08 00 06 04 00 01 36 15 FD 2A EE A3 C0 A8 08 64  | ...........]..........6.

                                                                                                                                           addr=2097152 | 00 00 00 00 00 00 D8 3A D4 64 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __  | ......


Writing = 0x915ff800
length = 42
addr=2099200 | 01 01 01 01 01 01 04 01 03 02 12 5D 08 06 00 01 08 00 06 04 00 01 36 15 FD 2A EE A3 C0 A8 08 64  | ...........]..........6.

                                                                                                                                           addr=2099200 | 00 00 00 00 00 00 D8 3A D4 64 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __  | ......


idx = 1026
```