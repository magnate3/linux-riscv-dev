
# mlx5dv

`mlx5dv_qp_init_attr`   mlx5dv_create_qp    mlx5dv_wr_set_mkey_sig_block   

# BUILD
=====

$ make
cc -std=c99 -Wall -g   -c -o pipelining_example.o pipelining_example.c
cc  -o ibv_pipelining_example pipelining_example.o -lmlx5 -libverbs -lrdmacm


EXAMPLES
========

# Run the server
```
./ibv_pipelining_example -S  10.22.116.221 -p 6666
 -----------------Configuration------------------
 Local IP : 10.22.116.221
 port : 6666
 Block size : 512
 I/O size : 4096
 Queue depth : 8
 ------------------------------------------------

Polls 5477, completions 30, comps/poll 0.0
Busy time 389986 ns, test time 1249371 ns, busy percent 0.04, time/comp 12999 ns
```


# Run the client
```
./ibv_pipelining_example -c  10 -p 6666 -l 2  10.22.116.221
 -----------------Configuration------------------
 Remote IP : 10.22.116.221
 port : 6666
 Block size : 512
 I/O size : 4096
 Queue depth : 8
 ------------------------------------------------

DEBUG: REP[0]: received with status SIG_ERROR
IOps : 0
```


# Run './ibv_pipelining_example --help' for more information.
