                            README
            GPUDirect enhancements for libibverbs utils
                          July 2020


Introduction
=============

The package includes the enhanced ibv_rc_pingpong and ibv_ud_pingpong tests to
use gpu buffers for data transfer. The base code is taken from libibverbs
examples of rdma-core-22.6 and added the options to enable gpu memory support

Build Instructions
==================

Note: cmake is required to build this package

1. Clone this repositoy
2. cd gpu_ibv_uitls
3. cmake .
4. cmake --build .

not use gpu

```
cmake -DUSE_HOST_MEM=on
```

This generates teh executables as rc_pingpong and ud_pingpong

Commands Help
============

- Run rc_pingpong with hostmemory
    - On Server
         ./rc_pingpong -d <ib_dev>  -g 3
    - On Client
         ./rc_pingpong -d <ib_dev>  -g 3 <server ip>

- Run rc_pingpong with gpu memory
    - On Server
         ./rc_pingpong -d <ib_dev>  -g 3 -C
    - On Client
         ./rc_pingpong -d <ib_dev>  -g 3 -C <server ip>

- Run uc_pingpong with hostmemory
    - On Server
         ./uc_pingpong -d <ib_dev>  -g 3
    - On Client
         ./ud_pingpong -d <ib_dev>  -g 3 <server ip>

- Run ud_pingpong with gpu memory
    - On Server
         ./ud_pingpong -d <ib_dev>  -g 3 -C
    - On Client
         ./ud_pingpong -d <ib_dev>  -g 3 -C <server ip>

- Verify the data in above experiments using -c option
- Check the help for all other options
# gpu_ibv_utils