# 深入浅出 llama.cpp（三）：拆解 simple.cpp —— 掌握推理的核心逻辑

## 零、前言

在上一篇文章中，成功搭建了开发环境并编译了第一个程序。今天，我将深入 `llama.cpp` 官方提供的最简示例—— `simple.cpp`。

虽然它只有 200 多行代码，但它完整地展示了一个大语言模型从加载到生成文字的“生命周期”。理解了这个文件，就掌握了 `llama.cpp` 的核心 API 调用规范。

## 一、核心步骤概览

阅读 `simple.cpp` 后，我们可以将推理过程抽象为以下流程：

1. **Backend 初始化**：让程序知道该用 CPU 还是 GPU 推理。
2. **Model 加载**：从硬盘加载 GGUF 权重。
3. **Tokenization**：将人类的文字变成模型能懂的数字（Token）。
4. **Context 创建**：为当前对话开辟一段“内存空间”和“上下文窗口”。
5. **Sampler 初始化**：决定模型如何从概率中“挑选”下一个字。
6. **Decode 循环**：最核心的环节，循环生成 Token 直到结束。

## 二、核心代码详解

### 1. 后端与模型加载

代码最开始调用了 `ggml_backend_load_all()`，这是现代 `llama.cpp` 的标配，用于自动加载所有可用的计算后端（CUDA、Metal 等）。

```cpp
// 初始化模型参数
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = ngl; // 决定有多少层在 GPU 跑

// 加载模型权重
llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
```
> 注意：`llama_model` 对象包含了庞大的模型权重，它是 **只读** 且 **线程安全** 的。

### 2. Tokenization (分词)
`llama.cpp` 的 API 设计非常硬核，获取 Prompt 的 Token 序列 **需要调用两次 `llama_tokenize`**：第一次为了 **获取长度**，第二次才真正 **填充数据**。

```cpp
// 第一次：获取 Token 数量（传入 NULL）
const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), ..., NULL, 0, ...);

// 第二次：根据长度分配空间并填充
std::vector<llama_token> prompt_tokens(n_prompt);
llama_tokenize(vocab, prompt.c_str(), ..., prompt_tokens.data(), prompt_tokens.size(), ...);
```

> 注意：`llama_tokenize()` 在 `tokens == nullptr` 或容量不足时，用“负返回值”表示 **需要的 token 数量**。
> 这是一个刻意的设计，正常情况的分词成功就以正数表示；
> 在代码中，由于只想得知所需 Token 数量而 **不需要进行分词** 传入了 NULL，并标明了没有缓冲区，
> 因此这一返回值必然为负值，需要对其取反。

### 3. 实例化上下文 (Context)

如果说 `llama_model` 是硬盘里的书，那么 `llama_context` 就是在读这本书时的“草稿纸”。

Context 决定了模型读取的总窗口大小和处理的 Token 数，用于 **配置模型的初始化参数**。

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = n_prompt + n_predict; // 总窗口大小
ctx_params.n_batch = n_prompt;           // 一次最多处理多少个 Token

llama_context *ctx = llama_init_from_model(model, ctx_params);
```

### 4. 采样器 (Sampler)

模型输出的是一组 **概率分布**，而“采样器”决定了我们是永远选概率最高的字（Greedy），还是带点随机性（Temperature）。

`simple.cpp` 演示了如何 **链式组合采样器**：

```cpp
auto sparams = llama_sampler_chain_default_params();
llama_sampler *smpl = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_greedy()); // 使用贪婪采样
```

### 5. Decode 循环

这是模型真正“思考”的地方。

1. **预处理 Prompt**：首先将整个输入的 Prompt 包装成一个 `batch` 送入 `llama_decode`。
2. **生成循环**：
    - 调用 `llama_decode` 进行计算。
    - 使用 `llama_sampler_sample` 从结果中选出一个 `new_token_id`。
    - 将生成的 `new_token_id` 转换回文字（Piece）打印。
    - 将这个新的 Token 再次包装成 `batch`，送入下一次迭代。

```cpp
for (int n_pos = 0; ...) {
    // 推理当前批次
    llama_decode(ctx, batch); 

    // 采样下一个 Token
    new_token_id = llama_sampler_sample(smpl, ctx, -1);
    
    // 转换文字并打印
    char buf[128];
    llama_token_to_piece(
        vocab, new_token_id, 
        buf, sizeof(buf), 
        ...);
    printf("%s", buf);

    // 准备下一次循环：将刚生成的字作为下一次的输入
    batch = llama_batch_get_one(&new_token_id, 1);
}
```

## 三、 为什么 `simple.cpp` 对二次开发很重要？

通过阅读这份源码，我发现了几个非常有价值的设计点：

- **RAII 的缺失**：你会发现代码最后有大量的 `_free` 函数。在 C++ 二次开发时，我们应该用 `std::unique_ptr` 封装这些资源（当然，这其实已经被官方封装在了 `include/llama-cpp.h` 中）
- **Batch 的灵活性**：`llama_batch_get_one` 只是冰山一角。对于高并发后端，我们需要手动构建 `llama_batch` 结构体，实现多序列并行推理。
- **流式输出**：代码中的 `fflush(stdout)` 提醒我们，推理是一个异步、持续的过程，这正是实现“打字机效果”的关键。

## 四、实战构建：试着进行一次模型调用！

### 1. 下载 huggingface_hub

`huggingface_hub` 包含有 HuggingFace（模型托管网站）的模型下载工具，能够简化这一过程。

```bash
pip install -U huggingface_hub
```

### 2. 下载小模型

这里选取业界比较出名的 Qwen2.5 模型，且为了降低性能开销，选取 0.5B 的小版本。

利用刚刚安装好的下载工具进行下载：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF --local-dir ./models --include qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 3.（可选）配置 HuggingFace 镜像源

如果安装卡住，可以配置镜像站链接的环境变量进行下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> 注：直接使用该环境变量配置只会在当前终端内生效，如果想要持久化则需要加入 `~/.bashrc`，
> 保存后执行 `source ~/.bashrc` 即可。

### 4. 构建并运行

仍然与上一篇博客一样，在构建目录下使用 CMake 进行构建。

```bash
cmake ..
cmake --build . --config Release
```

构建完成后，在 **项目根目录** 下执行即可看到结果：

```bash
./build/main3
# ...
Hello my name is Sarah and I am a 17 year old female.
```

## 五、结语

`simple.cpp` 像是一把钥匙，帮我们打开了底层 API 的大门。虽然它不支持多轮对话（没有处理 KV Cache 的移位），也不支持并发，但它为我们提供了最干净的骨架。

如果你用的是 CPU 进行推理，你会惊讶地发现这个模型跑得也太慢了！哪怕它只有 0.5B 的参数量（作为参考，Deepseek 满血版有 700 多 B），它也是一个字一个字往外蹦的。

这显然不是一个最优的做法，在下一篇博客中我会介绍 `llama.cpp` 的灵魂————KV Cache（键值缓存）管理。它能够有效解决现在遇到的性能问题！