```
nvcc gpuvm.cpp page_allocator.cpp -o gpuvm -lcuda
```
```
gpuvm-demo# ./gpuvm 
Reserved virtual memory range of 2MiB from 0x302000000 to 30a000000 this is a device virtual address space
Reserving 1 page of uvm space at 0x5d1cabd0aa90 this is a uvm address.
Freed page at 0x5d1cabd0aa90
Freed device virtual memory range.
```
