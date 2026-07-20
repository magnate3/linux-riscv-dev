## build

    gcc 01_single_process_multi_device.c --output build/01_single_process_multi_device -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lnccl -L/usr/local/cuda/lib64 -lcuda -lcudart

