
#  bug1

```
 ggml_backend_tensor_set(model.c, matrix_C, 0, ggml_nbytes(model.c));
```
引发GGML_ASSERT(buf != NULL && "tensor buffer not set") failed   

```
./build/simple-backend3
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 3090)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Core(TM) i9-14900)
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 0.00 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.00 MiB to 0.00 MiB
Handling Override Tensors for backends: CUDA0 CPU /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:264: GGML_ASSERT(buf != NULL && "tensor buffer not set") failed
/pytorch/GGML-Tutorial/build/ggml/src/libggml-base.so(+0x5ef21)[0x7cef680f5f21]
/pytorch/GGML-Tutorial/build/ggml/src/libggml-base.so(ggml_print_backtrace+0x281)[0x7cef680f61db]
/pytorch/GGML-Tutorial/build/ggml/src/libggml-base.so(ggml_abort+0x117)[0x7cef680f6351]
/pytorch/GGML-Tutorial/build/ggml/src/libggml-base.so(ggml_backend_tensor_set+0xbd)[0x7cef6810df59]
./build/simple-backend3(_Z7computeR12simple_modelP11ggml_cgraph+0xcf)[0x647f48f72f6e]
./build/simple-backend3(main+0x65)[0x647f48f7307b]
/usr/lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x7cef631d4d90]
/usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80)[0x7cef631d4e40]
./build/simple-backend3(_start+0x25)[0x647f48f728e5]
Aborted (core dumped)
```






```
(gdb) bt
#0  ggml_backend_graph_compute_async (backend=0x5555558df8e0, cgraph=0x555555933468) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:334
#1  0x00007ffff7eb459d in ggml_backend_sched_compute_splits (sched=0x555555930270) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1404
#2  0x00007ffff7eb51e3 in ggml_backend_sched_graph_compute_async (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1596
#3  0x00007ffff7eb5156 in ggml_backend_sched_graph_compute (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1580
#4  0x000055555556a10e in compute(simple_model&, ggml_cgraph*) ()
#5  0x000055555556a3ed in main ()
(gdb) 
(gdb) b ggml_backend_cpu_graph_compute
```


```
static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
}
```

```
ggml_compute_forward_mul_mat 
```

> ##   ggml_backend_cuda_graph_compute


```
(gdb) bt
#0  ggml_backend_cuda_graph_compute (backend=0x5555558df8e0, cgraph=0x555555933468) at /pytorch/GGML-Tutorial/ggml/src/ggml-cuda/ggml-cuda.cu:2710
#1  0x00007ffff7eb04fc in ggml_backend_graph_compute_async (backend=0x5555558df8e0, cgraph=0x555555933468) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:334
#2  0x00007ffff7eb459d in ggml_backend_sched_compute_splits (sched=0x555555930270) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1404
#3  0x00007ffff7eb51e3 in ggml_backend_sched_graph_compute_async (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1596
#4  0x00007ffff7eb5156 in ggml_backend_sched_graph_compute (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1580
#5  0x000055555556a196 in compute(simple_model&, ggml_cgraph*) ()
#6  0x000055555556a475 in main ()
(gdb) 
```

+   ggml_backend_buft_is_cuda_split
```
(gdb) bt
#0  ggml_backend_buft_is_cuda_split (buft=0x7ffff7f48b60 <ggml_backend_cpu_buffer_type::ggml_backend_cpu_buffer_type>) at /pytorch/GGML-Tutorial/ggml/src/ggml-cuda/ggml-cuda.cu:956
#1  0x00007ffff32e7520 in ggml_backend_cuda_device_supports_buft (dev=0x5555558de0c0, buft=0x7ffff7f48b60 <ggml_backend_cpu_buffer_type::ggml_backend_cpu_buffer_type>)
    at /pytorch/GGML-Tutorial/ggml/src/ggml-cuda/ggml-cuda.cu:3277
#2  0x00007ffff7eb0ccd in ggml_backend_dev_supports_buft (device=0x5555558de0c0, buft=0x7ffff7f48b60 <ggml_backend_cpu_buffer_type::ggml_backend_cpu_buffer_type>)
    at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:500
#3  0x00007ffff7eb0556 in ggml_backend_supports_buft (backend=0x5555558df8e0, buft=0x7ffff7f48b60 <ggml_backend_cpu_buffer_type::ggml_backend_cpu_buffer_type>)
    at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:342
#4  0x00007ffff7eb121a in ggml_backend_sched_backend_from_buffer (sched=0x555555930270, tensor=0x7fffc966f060, op=0x7fffc966f060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:705
#5  0x00007ffff7eb12f1 in ggml_backend_sched_backend_id_from_cur (sched=0x555555930270, tensor=0x7fffc966f060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:732
#6  0x00007ffff7eb1cc7 in ggml_backend_sched_split_graph (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:890
#7  0x00007ffff7eb5106 in ggml_backend_sched_alloc_graph (sched=0x555555930270, graph=0x7fffc88f2060) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1568
#8  0x0000555555569fb3 in compute(simple_model&, ggml_cgraph*) ()
#9  0x000055555556a475 in main ()
(gdb) 
```

> ## debug

```
 env GGML_SCHED_DEBUG=2  ./build/simple-backend3 
```

# offload LLAMA_SUPPORTS_GPU_OFFLOAD