
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

```
 env GGML_SCHED_DEBUG=2  ./build/simple-backend3 
```

#  调度器优先级


+  ggml_backend_buffer_set_usage(buf2, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```
   print_buft(buf1);
     ggml_backend_buffer_set_usage(buf1, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
     //model.c = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32,  rows_A,cols_B);
     model.c = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols_C, rows_C);
     model.d = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, cols_C, rows_C);
     //model.d = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, rows_C, cols_C);
     //model.c = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, 1);
     ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(cpu_ctx, model.cpu_buft);
     if(buf2 == nullptr)
     {
             throw std::runtime_error(("unable to allocate buffer2"));
     }
     //backend_bufts2.emplace_back(buf1);
     //backend_bufts2.emplace_back(buf2);
     //model.sched = ggml_backend_sched_new(backends2,backend_bufts2.data(), 2, 32, false, true);
     ggml_backend_buffer_set_usage(buf2, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

```

 !!!!!!! node mul_mat_0 on  device CUDA0 

 !!!!!!! node add_0 on  device CPU 

 !!!!!!! node add_1 on  device CPU 

 !!!!!!! backend  on  device CUDA0 

 !!!!!!! backend  on  device CPU 
+++++++++++ splits: 2
```
此时splits: 2，cpu上有计算节点



+  不设置 ggml_backend_buffer_set_usage(buf2, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

```

 !!!!!!! node mul_mat_0 on  device CUDA0 

 !!!!!!! node add_0 on  device CUDA0 

 !!!!!!! node add_1 on  device CUDA0 

 !!!!!!! backend  on  device CUDA0 

 !!!!!!! backend  on  device CPU 
+++++++++++ splits: 1


```

此时splits: 1，cpu上没有有计算节点，计算节点都在CUDA0 

>  ##  cudaMemcpyAsync


```

 !!!!!!! node mul_mat_0 on  device CUDA0 

 !!!!!!! node add_0 on  device CUDA0 

 !!!!!!! node add_1 on  device CUDA0 

 !!!!!!! backend  on  device CUDA0 

 !!!!!!! backend  on  device CPU 
+++++++++++ splits: 1
```

```
(gdb) bt
#0  0x00007ffff2a71fa4 in cudaMemcpyAsync () from /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
#1  0x00007ffff32dc98e in ggml_backend_cuda_buffer_set_tensor (buffer=0x555556203aa0, tensor=0x5555558dd1a0, data=0x555555579020 <matrix_A>, offset=0, size=32)
    at /pytorch/GGML-Tutorial/ggml/src/ggml-cuda/ggml-cuda.cu:581
#2  0x00007ffff7d89015 in ggml_backend_tensor_set (tensor=0x5555558dd1a0, data=0x555555579020 <matrix_A>, offset=0, size=32) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:268
#3  0x000055555556a2c6 in compute(simple_model&, ggml_cgraph*) ()
#4  0x000055555556a78b in main ()
(gdb) 
```
ggml_backend_tensor_copy --> gml_backend_tensor_set    
```
(gdb) bt
#0  0x00007ffff2a71fa4 in cudaMemcpyAsync () from /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
#1  0x00007ffff32dc98e in ggml_backend_cuda_buffer_set_tensor (buffer=0x5555564703f0, tensor=0x7fffebdcb5f0, data=0x555556470740, offset=0, size=48)
    at /pytorch/GGML-Tutorial/ggml/src/ggml-cuda/ggml-cuda.cu:581
#2  0x00007ffff7d89015 in ggml_backend_tensor_set (tensor=0x7fffebdcb5f0, data=0x555556470740, offset=0, size=48) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:268
#3  0x00007ffff7d896f8 in ggml_backend_tensor_copy (src=0x7fffc966f1d0, dst=0x7fffebdcb5f0) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:378
#4  0x00007ffff7d8d550 in ggml_backend_sched_compute_splits (sched=0x555555930560) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1398
#5  0x00007ffff7d8e1e3 in ggml_backend_sched_graph_compute_async (sched=0x555555930560, graph=0x5555562dfba0) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1596
#6  0x00007ffff7d8e156 in ggml_backend_sched_graph_compute (sched=0x555555930560, graph=0x5555562dfba0) at /pytorch/GGML-Tutorial/ggml/src/ggml-backend.cpp:1580
#7  0x000055555556a519 in compute(simple_model&, ggml_cgraph*) ()
#8  0x000055555556a78b in main ()
(gdb) c
```

> ## ggml_backend_buffer_is_host



```
    if(ggml_backend_buffer_is_host(model.a->buffer))
    {
          printf("model.a is host buffer\n");
    }
    if(ggml_backend_buffer_is_host(model.c->buffer))
    {
          printf("model.c is host buffer\n");
    }
model.c is host buffer
```

> ## GGML_OP  GGML_OP_MUL_MAT_ID


```
// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                int n_embd_head = hparams.n_embd_head_v;
                int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                // FIXME
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12345, w->ne[1], 6789);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // FIXME
                const int64_t d_state      = w->ne[0];
                const int64_t d_inner      = w->ne[1];
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 1;
                ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, d_inner, n_seqs);
                ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * B = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                ggml_tensor * C = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd = hparams.n_embd;
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

```