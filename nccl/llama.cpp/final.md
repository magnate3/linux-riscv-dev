## 实验报告

### 实验目的和实验主要内容

在大模型推理过程中，KV Cache（Key-Value Cache）被广泛用于存储中间计算结果，以减少重复计算开销并提升推理效率。然而，随着模型推理上下文长度的增加，KV Cache的规模急剧扩大，导致GPU显存资源紧张，甚至可能引发模型推理性能下降或无法运行的问题。此外，通过对KV Cache的分析发现，模型推理过程中KV Cache的稀疏性显著，即并非所有存储的KV对都需要参与后续计算。这为优化KV Cache的存储和使用提供了可能性。

基于上述背景，本实验探索将KV Cache部分存储在CPU内存中，仅在推理时动态加载必要的KV对到GPU进行计算。通过这种方式，可以在缓解GPU显存压力的同时，尽可能减少对推理延迟的影响。

本实验旨在复现InfLLM的KV Cache管理方案，结合动态加载策略，优化KV Cache的选取和加载过程。具体目标包括：

1. 针对大模型推理场景，KV Cache的分离存储，将部分KV对存储在CPU内存中，以缓解GPU显存紧张问题；
2. 优化KV Cache的动态加载策略，确保在降低显存占用的同时，尽可能减少对推理延迟的影响；
3. 基于llama.cpp框架，实现InfLLM的KV Cache高效分离方案，并验证其在显存占用和推理效率方面的优化效果。

在llama.cpp中复刻的意义在于，首先llama.cpp的运行速度更快。另外llama.cpp提供了很多后端，而且基于cpp能够在不同平台上编译，便于在包括嵌入式设备在内的各类设备上部署。原有的InfLLM是基于pytorch的，跨平台时难度较大，对于很多平台，可能运行pytorch本身会造成较大开销。

### 实验内容

#### 前期准备工作

自行进行了实现环境搭建，这部分消耗了一定时间。

在NVIDIA RTX 4090上使用CUDA 12.8进行实验。

- InfLLM的实验环境安装

    KV cache的offload相关的3篇文献中，InfLLM是给了明确代码的，所以更容易复现结果，以更加方便移植到llama.cpp中。

    不过InfLLM的环境比较老，使用最新版的各个包会出现错误。

    - 复现结果时，出现了如下bug。加入PYTHONPATH=.的环境变量后解决

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 9, in <module>
            from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
        ModuleNotFoundError: No module named 'inf_llm'
        ```

    - 然后又出现了huggingface要求登录的提示

        加入HUGGING_FACE_HUB_TOKEN的环境变量后解决

    - 然后出现如下报错：

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 289, in <module>
            model, tokenizer = get_model_and_tokenizer(args.model)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 56, in get_model_and_tokenizer
            model = patch_hf(model, config.type, **config)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/inf_llm/utils/patch.py", line 152, in patch_hf
            hf_rope = model.model.layers[0].self_attn.rotary_emb 
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的家目录>/miniconda3/envs/infllm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1928, in __getattr__
            raise AttributeError(
        AttributeError: 'MistralAttention' object has no attribute 'rotary_emb'
        ```

        猜测是版本不同导致rotary_emb没有了，于是检索到一个类似问题

        结合 [AttributeError: 'LlamaAttention' object has no attribute 'rotary_emb'](https://github.com/unslothai/unsloth/issues/1443) 中的回答，执行如下代码，改变环境配置

        ```bash
        pip install unsloth && pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps https://github.com/unslothai/unsloth/archive/refs/tags/November-2024.zip
        pip uninstall tokenizers
        pip uninstall transformers
        pip install tokenizers==0.20.3 transformers==4.46.1
        ```

        如此操作，解决了如上报错


- llama.cpp的实验环境安装

    服务器上存在多个版本的nvcc，需要手动指定nvcc编译器。通过以下指令指定
    ```bash
    export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
    ```

    服务器上存在多个GPU计算节点，而家目录只是挂载到每个计算节点的同一位置。主节点上没有GPU，所以只能在计算节点编译，不过没有计算节点的root权限，所以自行编译cmake等工具链。

    - 手动编译cmake
    
        ```
        wget https://github.com/Kitware/CMake/releases/download/v4.0.0/cmake-4.0.0.tar.gz
        tar axf cmake-4.0.0.tar.gz
        cd cmake-4.0.0
        ./bootstrap
        ./configure --prefix={我的安装目录}/cmake
        make
        make install

        export PATH={我的安装目录}/cmake/bin:$PATH
        ```

        但是在上面安装中遇到了缺少OpenSSL的报错，所以需要自行编译OpenSSL，然后设置OPENSSL_ROOT_DIR环境变量后解决

        ```
        wget https://github.com/openssl/openssl/releases/download/openssl-3.4.1/openssl-3.4.1.tar.gz
        tar axf openssl-3.4.1.tar.gz
        cd openssl-3.4.1
        ./config --prefix={我的安装目录}/openssl
        make
        make install

        export OPENSSL_ROOT_DIR={我的安装目录}/openssl
        ```


    - 编译后工具链后，编译llama.cpp的CUDA版本：

        ```
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release -j
        ```

        然后根据https://qwen.readthedocs.io/zh-cn/latest/run_locally/llama.cpp.html，从hugging face上下载了qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf这个模型，通过以下指令在CLI中运行这个模型。

        ```
        ./llama-cli -m qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
            -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
            -fa -ngl 80 -n 512
        ```

#### InfLLM优化的主要原理

`context_manager.py`中的`ContextManager`是KV cache相关的主要代码。InfLLM中将kv分成init, global, local三组。对于global部分的kv，使用分组的方式组织，并且在每个分组中保留重要的token的kv，同时记录local_score。`ContextManager`以`MemoryUnit`为单位管理张量将kv cache切分为块卸载到CPU上，而GPU上只保留每个块中有代表性的几个k，以作查询。推理时，结合local_score确定相应块的重要程度，决定是否驻留在显存中。

#### llama.cpp代码分析

llama.cpp项目本身非常复杂，核心代码较多。其前后端分离的设计，使得对于新增的异构设备，只需要提供相应接口，并且能够执行对静态图的计算即可，可以有效兼容各种异构设备。并且cpp的代码能够在大多数平台上编译，并且开销更低。cpp中对于内存管理、同步互斥等的支持更多，可以更方便管理硬件资源，尤其是嵌入式设备等硬件资源有限的设备上。

不过llama.cpp的这种设计导致代码量增加，进行后端抽象以及静态计算图造成了debug和代码阅读的一些麻烦，尤其对于没有经验者。

llama.cpp中代码比较多，有几万行代码，网上能搜到一些参考资料，不过经过llama.cpp的几次重构和功能增加，很多网上的资料和我实际看到的接口并不完全一致。有很多内容，需要手动输出，然后结合代码，猜测其含义。

- 前后端交互方式（以CUDA为例，其他类似）

    `ggml/src/ggml-*`中是各种后端实现和一个`CMakeLists.txt`，主要包括算子和前后端数据传输。

    在cuda后端中有详细的实现，不过有些后端只是对cuda进行简单的函数名替换。比如对于hip后端，在`ggml/src/ggml-cuda/vendors/hip.h`中定义了如下宏（片段）

    ```C++
    ...
    #define cublasCreate hipblasCreate
    #define cublasDestroy hipblasDestroy
    #define cublasGemmEx hipblasGemmEx
    ...
    ```

    - 数据传输部分是通过传递一系列函数指针实现的。
    
        比如`ggml/src/ggml-cuda/ggml-cuda.cu`中的
        ```C++
        static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
        ```

        然后通过下面这个结构体，将函数入口传递到ggml_backend_cuda_buffer_interface的接口中
        ```C++
        static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
            /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
            /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
            /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
            /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
            /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
            /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
            /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
            /* .clear           = */ ggml_backend_cuda_buffer_clear,
            /* .reset           = */ NULL,
        };
        ```
        
        在前端中，这个接口的定义在`ggml/src/ggml-backend-impl.h`如下
        ```C++
        //
        // Backend buffer
        //

        struct ggml_backend_buffer_i {
            // (optional) free the buffer
            void         (*free_buffer)  (ggml_backend_buffer_t buffer);
            // base address of the buffer
            void *       (*get_base)     (ggml_backend_buffer_t buffer);
            // (optional) initialize a tensor in the buffer (eg. add tensor extras)
            enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
            // tensor data access
            void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
            void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
            void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
            // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
            bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
            // clear the entire buffer
            void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
            // (optional) reset any internal state due to tensor initialization, such as tensor extras
            void         (*reset)        (ggml_backend_buffer_t buffer);
        };
        ```

        设计原理类似于把后端当成一种库，然后把各个调用的入口通过这种方式告知前端。

        讲过包装，实现了`ggml_backend_tensor_set`，调用这个函数可以将数据设定到后端的tensor中，将接口的复杂性保留在实现内部（也方便改接口）。

        ```C++
        void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            GGML_ASSERT(tensor);
            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

            if (size == 0) {
                return;
            }

            GGML_ASSERT(buf != NULL && "tensor buffer not set");
            GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

            buf->iface.set_tensor(buf, tensor, data, offset, size);
        }
        ```

    - 后端各种算子的调用和实现

        这种后端算子的实现，感觉和操作系统中的系统调用中的一些设计比较像（不过当然没有特权级的切换）。把后端作为一种类似“操作系统”的支持，然后前端通过特定的调用号来调用相应操作。不同的后端都约定了相同的OP_CODE和调用方式以便前端可忽略后端的复杂性。

        因为静态计算图，不能直接调用后端函数。所以将需要计算的张量所需要的操作通过GGML_OP_CODE的方式记录，

        ```C++
        // available tensor operations:
        enum ggml_op {
            GGML_OP_NONE = 0,

            GGML_OP_DUP,
            GGML_OP_ADD,
            ...
        }
        ```

        然后比如这个算子，通过给OP_CODE和OP_PARAMS记录下所需要进行的调用。

        （不知道操作系统中有无类似的将很多个系统调用记录下来，然后合并进行调用的方式，以减少系统调用开销？）

        ```C++
        struct ggml_tensor * ggml_abs(
                struct ggml_context * ctx,
                struct ggml_tensor  * a) {
            return ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
        }
                
        struct ggml_tensor * ggml_unary(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_unary_op    op) {
            return ggml_unary_impl(ctx, a, op, false);
        }

        static struct ggml_tensor * ggml_unary_impl(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_unary_op    op,
                bool                  inplace) {
            GGML_ASSERT(ggml_is_contiguous_1(a));

            struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

            ggml_set_op_params_i32(result, 0, (int32_t) op);

            result->op     = GGML_OP_UNARY;
            result->src[0] = a;

            return result;
        }

        static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
            assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
            ((int32_t *)(tensor->op_params))[i] = value;
        }
        ```

        然后后端`ggml/src/ggml-cuda/ggml-cuda.cu`调用`ggml_cuda_compute_forward`识别所需要进行的计算。
        ```C++
        static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
            switch (dst->op) {
                case GGML_OP_ARGMAX:
                    ggml_cuda_argmax(ctx, dst);
                    break;
                case GGML_OP_COUNT_EQUAL:
                    ggml_cuda_count_equal(ctx, dst);
                    break;
                ...
            }
        }
        ```

#### InfLLM在llama.cpp中的复现

<!-- InfLLM文章中就提到了在llama.cpp中复现，不过并没有实现。这和llama.cpp和InfLLM的底层数据结构和算法设计不同，导致完全复现实验较为困难。 -->

**代码见本仓库的src/llama.cpp部分**

复现时还是遇到了比较多的问题，比如llama.cpp中算子支持不全，有的算子需要特定条件下才能使用，否则会出现数据不连续、类型不一致等报错信息，比如：

```
llama.cpp/ggml/src/ggml.c:2040: GGML_ASSERT(ggml_can_repeat(b, a)) failed
llama.cpp/ggml/src/ggml-cuda/sumrows.cu:33: GGML_ASSERT(ggml_is_contiguous(src0)) failed
```

- 建立静态计算图

    - qwen2对应的应该是`llama_model::build_graph`中应该是调用`LLM_ARCH_QWEN2`的case。需要定位到核心代码如下：

        ```c++
        struct llm_build_qwen2 : public llm_graph_context {
            llm_build_qwen2(const llama_model & model, const llm_graph_params & params, ggml_cgraph * gf) : llm_graph_context(params) {
                ...
                for (int il = 0; il < n_layer; ++il) {
                    ggml_tensor * inpSA = inpL;

                    // norm
                    cur = build_norm(inpL,
                            model.layers[il].attn_norm, NULL,
                            LLM_NORM_RMS, il);
                    cb(cur, "attn_norm", il);

                    // self-attention
                    {
                        // compute Q and K and RoPE them
                        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                        ...
                        cur = build_attn(inp_attn, gf,
                                model.layers[il].wo, model.layers[il].bo,
                                Qcur, Kcur, Vcur, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                    }
                    ...
                }
                ...
            }
        };
        ```

        然后定位到需要修改的是attention相关的计算图，在这部分中加入representative相关的逻辑（对于evited部分的token的k和后面若干个q做内积取平均，作为r，然后取top_k作为representative tokens）：

        ```C++
        ggml_tensor * llm_graph_context::build_attn(
                llm_graph_input_attn_kv_unified * inp,
                ggml_cgraph * gf,
                ggml_tensor * wo,
                ggml_tensor * wo_b,
                ggml_tensor * q_cur,
                ggml_tensor * k_cur,
                ggml_tensor * v_cur,
                ggml_tensor * kq_b,
                    float     kq_scale,
                    int       il) const {
            // these nodes are added to the graph together so that they are not reordered
            // by doing so, the number of splits in the graph is reduced
            ...
        }
        ```

        计算score的时候遇到的一些问题有：

        因为我下载调试的模型使用了混合精度，所以有些张量是float16的，但是ggml的算子支持不是很全面，有些算子只能计算float32，或者有其他限制条件。进行计算的张量的维度也是固定的，所以为了适应算子，需要将需要计算的维度调整到算子中指定的维度，有时还需要进行维度合并才能满足算子的维度要求。

        另外由于float16的范围比较有限，有些运算可能出现溢出，再经过一些算子后，最后结果出现nan。而且由于是静态图，不能在计算是判断每个算子的结果是否有nan，只能将中间过程记录，等待计算图完成计算后再输出进行debug。
        
        解决了核心代码如下：

        ```C++
        auto k_need_score_num = n_tokens - score_block_size + 1;
        k_need_score_num = (k_need_score_num/score_block_size)*score_block_size;
        auto k_cur_to_score = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        ggml_tensor * k_need_score = ggml_view_3d(ctx0, k_cur_to_score,
            n_embd_head_k, k_need_score_num, n_head_kv,
            k_cur_to_score->nb[1],
            k_cur_to_score->nb[2],
            0);
        ggml_tensor * kq_need_score = ggml_mul_mat(ctx0, k_need_score, q);
        ggml_mul_mat_set_prec(kq_need_score, GGML_PREC_F32);
        ggml_tensor * score = ggml_view_3d(ctx0, kq_need_score,
            k_need_score_num, score_block_size, kq_need_score->ne[2],
            kq_need_score->nb[1]+ggml_row_size(kq_need_score->type, 1),
            kq_need_score->nb[2],
            0);
        score = ggml_view_2d(ctx0, score, k_need_score_num, score->ne[1]*score->ne[2], score->nb[1], 0);
        score = ggml_transpose(ctx0, score);
        score = ggml_mean(ctx0, score);
        score = ggml_transpose(ctx0, score);
        const_cast<llama_kv_cache_unified *>(kv_self)->score_valid_len[il] = k_need_score_num>0 ? n_tokens - score_block_size + 1 : 0;
        
        auto block_score = ggml_view_2d(ctx0, score, score_block_size, k_need_score_num/score_block_size, ggml_row_size(score->type, score_block_size), 0);
        auto block_score_f32 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, block_score->ne[0], block_score->ne[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, block_score, block_score_f32));
        auto score_top_k = ggml_top_k(ctx0, block_score_f32, representative_num);

        auto k_represent = ggml_view_4d(ctx0, k_need_score, k_need_score->ne[0], score_block_size, k_need_score->ne[1]/score_block_size, k_need_score->ne[2], k_need_score->nb[1], k_need_score->nb[1]*score_block_size, k_need_score->nb[2], 0);
        auto k_represent_perm = ggml_permute(ctx0, k_represent, 0, 2, 3, 1);
        auto k_represent_top_k = ggml_view_3d(ctx0, k_represent_perm, k_represent_perm->ne[0]*k_represent_perm->ne[1], k_represent_perm->ne[2], k_represent_perm->ne[3], k_represent_perm->nb[2], k_represent_perm->nb[3], 0);
        k_represent_top_k = ggml_get_rows(ctx0, k_represent_top_k, score_top_k);

        k_represent_top_k = ggml_view_4d(ctx0, k_represent_top_k, k_represent_perm->ne[0], k_represent_top_k->ne[0]/k_represent_perm->ne[0], k_represent_top_k->ne[1], k_represent_top_k->ne[2], k_represent_top_k->nb[1]/representative_num, k_represent_top_k->nb[1], k_represent_top_k->nb[2], 0);

        k_represent_top_k = ggml_permute(ctx0, k_represent_top_k, 1, 2, 0, 3);
        k_represent_top_k = ggml_mean(ctx0, k_represent_top_k);
        k_represent_top_k = ggml_view_3d(ctx0, k_represent_top_k, k_represent_top_k->ne[1], k_represent_top_k->ne[2], k_represent_top_k->ne[3], k_represent_top_k->nb[2], k_represent_top_k->nb[3], 0);
        k_represent_top_k = ggml_permute(ctx0, k_represent_top_k, 0, 2, 1, 3);
        auto k_represent_l_il = kv_self->k_represent_l[il];
        k_represent_l_il = ggml_view_3d(ctx0, k_represent_l_il, k_represent_top_k->ne[0], k_represent_top_k->ne[1], k_represent_top_k->ne[2], k_represent_top_k->nb[1], k_represent_top_k->nb[2], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_represent_top_k, k_represent_l_il));
        ```

- CPU和GPU间数据传输和同步

    虽然可以直接使用cudaMemcpyAsync和cudaStreamSynchronize进行数据传输和同步，但是这回破坏llama.cpp的前后端分离的设计，所以尽可能使用llama.cpp的接口。

    llama.cpp中提供了若干接口是：
    ```C++
    void ggml_backend_tensor_set_async(ggml_backend_t backend, ggml_tensor *tensor, const void *data, size_t offset, size_t size);
    void ggml_backend_tensor_get_async(ggml_backend_t backend, const ggml_tensor *tensor, void *data, size_t offset, size_t size);
    void ggml_backend_synchronize(ggml_backend_t backend);
    ```

    但是在`llm_build_qwen2`这个类中无法直接访问`ggml_backend_t backend`，所以增加了一些成员，`ggml_backend_t *`指针传递到静态图建立过程的代码处。这样才能够读写相应的显存中的内容，完成GPU和CPU之间的数据卸载和装载。ggml库中，虽然每个张量都有类型，但是`ggml_backend_tensor_get_async`的offset和size都是指字节数量，所以编程时需要注意。

    GPU数据卸载的部分核心代码如下所示：
    ```C++
    if (kv_self->k_offload[il]==nullptr) {
        const_cast<llama_kv_cache_unified *>(kv_self)->k_offload[il] = new char[4096*128*16*ggml_type_size(k_cache_view->type)]; // may be fp16 stored
        const_cast<llama_kv_cache_unified *>(kv_self)->v_offload[il] = new char[4096*128*16*ggml_type_size(k_cache_view->type)];
    }
    // OFFLOAD
    ggml_backend_tensor_get_async(backend, k_cache_view, kv_self->k_offload[il], 0, k_offload_len*ggml_type_size(k_cache_view->type));
    ggml_backend_tensor_get_async(backend, v_cache_view, kv_self->v_offload[il], 0, k_offload_len*ggml_type_size(k_cache_view->type));
    synchronize();
    ```

    装载时，首先需要读取GPU计算出来的sim(Block, Local_Context)，如下所示：
    ```C++
    float* similarity=new float[num_blocks];
    ggml_backend_tensor_get_async(backend, kv_self->score_l[il], similarity, 0, num_blocks*sizeof(float));
    synchronize();
    ```

    然后根据这个相似度，比对最重要的block是哪些，然后判断哪些block目前没有在gpu上（只有不在GPU才会触发CPU和GPU间的数据传输，以减少数据传输）。核心代码片段如下所示

    ```C++
    auto represent_blocks_num = num_blocks-(initial_block_len/score_block_size)-(local_block_len/score_block_size);
    auto& last_gpu_idx = const_cast<llama_kv_cache_unified *>(kv_self)->gpu_load_idx[il];
    std::vector<int> curr_gpu_idx;
    curr_gpu_idx.resize(max_blocks_num, -1);
    if (kv_self->predict_len[il]==1) {
        last_gpu_idx.resize(max_blocks_num, -1);
        std::fill(last_gpu_idx.begin(), last_gpu_idx.end(), -1);
    }
    std::vector<std::pair<float, int>> sort_idx;
    for (int i=(initial_block_len/score_block_size);i<num_blocks-(local_block_len/score_block_size);++i) {
        sort_idx.emplace_back(similarity[i], i);
    }
    std::sort(sort_idx.begin(), sort_idx.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    });
    auto block_bytes = n_embd_head_k * score_block_size * n_head_kv * ggml_type_size(k_cache_view->type);
    for (int j=0; j<max_blocks_num; ++j) {
        curr_gpu_idx[j] = sort_idx[j].second;
    }
    for (int j=0; j<max_blocks_num; ++j) {
        auto it = std::find(curr_gpu_idx.begin(), curr_gpu_idx.end(), last_gpu_idx[j]);
        if (it != curr_gpu_idx.end()) {
            auto index = std::distance(curr_gpu_idx.begin(), it);
            if (index!=j) {
                std::swap(curr_gpu_idx[index], curr_gpu_idx[j]);
            }
        }
    }
    for (int j=0; j<max_blocks_num; ++j) {
        // LOAD when necessary only
        if (curr_gpu_idx[j]!=last_gpu_idx[j]) {
            ggml_backend_tensor_set_async(backend, k_cache_view, kv_self->k_offload[il]+(curr_gpu_idx[j]*block_bytes), block_bytes*j+(n_embd_head_k * initial_block_len * n_head_kv * ggml_type_size(k_cache_view->type)), block_bytes);
            ggml_backend_tensor_set_async(backend, v_cache_view, kv_self->v_offload[il]+(curr_gpu_idx[j]*block_bytes), block_bytes*j+(n_embd_head_k * initial_block_len * n_head_kv * ggml_type_size(k_cache_view->type)), block_bytes);
        }
    }
    synchronize();
    const_cast<llama_kv_cache_unified *>(kv_self)->gpu_load_idx[il] = curr_gpu_idx;
    ```

    如果想要确实优化了LOAD过程（只有数据不在GPU上时才会进行数据装载），可以将`src/llama-graph.cpp`的1494行解除注释，这样会在每次发生装载的时候输出装载的block的index。结合similarity中的值，可以观察到只有上下文变化的时候才会有数据传输。

- sim(Block, Local_Contex)的计算

    由于llama.cpp的特点，导致infllm的部分实现可能在llama.cpp中比较不方便，以及和llama.cpp的管理kv的方式不同。所以这里采用了一种折中的方式解决sim(Block, Local_Contex)的计算。对于预测第$t$个token的sim记为$sim[t]$，采用移动平均方式更新sim：$sim[t]=(1-\alpha)\cdot sim[t-1]+\alpha\cdot new\_sim$。这样可以保证每个token的sim确实只是其附近若干token的和block的相关性，而且所涉及的block也不会发生频繁变化。核心代码如下所示
    ```C++
    auto num_blocks = kv_self->score_valid_len[il]/score_block_size;
    auto k_represent_l_il = kv_self->k_represent_l[il];
    k_represent_l_il = ggml_view_3d(ctx0, k_represent_l_il, n_embd_head_k, num_blocks, n_head_kv, ggml_row_size(k_represent_l_il->type, n_embd_head_k), ggml_row_size(k_represent_l_il->type, n_embd_head_k)*num_blocks, 0);
    k_represent_l_il = ggml_clamp(ctx0, k_represent_l_il, -10, 10); // avoid overflow
    
    auto similarity = ggml_mul_mat(ctx0, k_represent_l_il, q);
    ggml_mul_mat_set_prec(similarity, GGML_PREC_F32);
    similarity = ggml_permute(ctx0, similarity, 1, 2, 0, 3);
    similarity = ggml_mean(ctx0, similarity);
    similarity = ggml_transpose(ctx0, similarity); // num_blocks
    ggml_tensor* similarity_il = ggml_view_1d(ctx0, kv_self->score_l[il], num_blocks, 0);
    if (kv_self->predict_len[il]) {
        similarity = ggml_mul(ctx0, similarity, ggml_arange(ctx0, 0.05, 0.5, 1.)); // *.05
        similarity = ggml_add(ctx0, similarity, ggml_mul(ctx0, similarity_il, ggml_arange(ctx0, 0.95, 1.5, 1.))); // + old*.95
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, similarity, similarity_il));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, similarity, similarity_il));
    }
    const_cast<llama_kv_cache_unified *>(kv_self)->predict_len[il]++;
    ```


### 总结

通过本次实验，了解了LLM推理时可能面临的问题，以及一些优化要点和量化分析手段，也了解了llama.cpp的框架和设计逻辑。同时对于环境配置和工具链编译等方面也有了一些经验。对于异构加速器的使用。

由于infllm和llama.cpp的设计本身差异很大（llama.cpp的kv-cache设计可能没有考虑对offload的支持），导致难以实现完全相同的效果。这里采取了一种折中的方式，将infllm的主要kv offload的设计思路在llama.cpp上进行了实现，具体细节上有一些调整。不可否认，这些调整可能影响了性能。不过想要在llama.cpp中实现更有效的kv-cache，首先可能ggml支持的算子需要更多，另外在设计上增加很多类的接口，便于进行数据卸载和装载。可能这两方面的改进是更有效的llama.cpp上的kv offload的前提。不过增加算子支持的工作量较大，而增加类的接口可能需要彻底修改很多类的设计，而这超出了本次大实验的范围，不过仍然是值得尝试的问题。

感谢陈渝老师、王拓为助教和郝子胥助教的指导。

本学期我事情较多，在时间安排上也给大家带来了一些困扰，感谢老师、助教以及同学的理解和支持。
