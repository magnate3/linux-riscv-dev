

```
  docker run -d --rm --net=host  --name rocket   --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch nvcr.io/nvidia/pytorch:24.05-py3
```
#  flash-attn
```
pip install -r requirements.txt
pip install flash-attn==2.6.3
pip install modelscope
pip install addict -i https://pypi.tuna.tsinghua.edu.cn/simple
```


> ## 

+ pytorch 
```
 python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
2.4.1+cu121
12.1
```


```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

PyTorch 期望的 CUDA 版本：12.1（来自 2.4.1+cu121）    
系统全局编译器的 CUDA 版本：12.4（来自 nvcc --version）    

PyTorch 是 cu121，需要强行指定 flash-attn 去官方针对 PyTorch 2.4 + CUDA 12.1 编译好的现成仓库（Wheel）中下载，避免本地触发错误的 nvcc 12.4 编译。按照以下顺序执行：

以下方法无效
```
# 1. 彻底卸载当前有问题的 flash-attn
pip uninstall flash-attn -y

# 2. 清理 pip 缓存防止重用错误版本
pip cache purge

# 3. 使用官方预编译源安装（会自动匹配你的 torch 和 cuda 版本）
pip install flash-attn --no-build-isolation --extra-index-url   https://pypi.tuna.tsinghua.edu.cn/simple

```
pytorch cuda 和nvcc 版本还是不统一
```
root@ubuntu:/pytorch# python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
2.4.1+cu121
12.1
root@ubuntu:/pytorch# nvcc
nvcc fatal   : No input files specified; use option --help for more information
root@ubuntu:/pytorch# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
root@ubuntu:/pytorch#

# data处理

local_modelscope_download_longbench.py    


# 运行脚本

scripts/longbench/qwen2.5-0.5b-instruct.sh    

# 移植支持qwen2

## config

```
cat config/pipeline_config/longbench/qwen2.5-0.5b-instruct.json
{
    "pipeline_params": {
        "model_name": "/pytorch/models/Qwen2___5-0___5B-Instruct",
        "tokenizer_name": "/pytorch/models/Qwen2___5-0___5B-Instruct",
        "chat_template": "qwen",
        "model_max_len": 127500,
        "fattn": false,
        "use_flash_attn": false,
        "truncation_mode": "middle",
        "batch_size": 1,
        "out_of_max_len_allowed": true,
        "base": 500000,
        "rope_theta_factor": 1.0
    }
}
```


##  python代码

+ pipeline/inf_stream_llm/inf_llm/infllm.py    

```
def initialize_model_tokenizer(pipeline_config)
```
+ pipeline/inf_stream_llm/main.py    

只测试longbench    

+  pipeline/model_utils.py

```
def build_chat(tokenizer, prompt, chat_template):
```

+  pipeline/inf_stream_llm/inf_llm/utils/patch.py
```

def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base = None,
    distance_scale = None,
    rope_theta_factor = None,
    rope_linear_scaling_factor = None,
    **kwargs
):
```

+  pipeline/inf_stream_llm/inf_llm/utils/greedy_search.py

更改  
```
def generate(self, text=None, input_ids=None, **kwargs):
```
添加
```
def get_pure_qwen(model_path)
```

# 结果

```
cat res/longbench/rocket/1234/qwen2.5-0.5b-instruct/longbench_result_summary.json
{
    "individual_dataset_result": {
        "narrativeqa": 10.76,
        "qasper": 23.78,
        "multifieldqa_en": 38.56,
        "hotpotqa": 23.11,
        "2wikimqa": 24.54,
        "musique": 15.14,
        "gov_report": 12.73,
        "qmsum": 18.96,
        "multi_news": 12.06,
        "trec": 64.0,
        "triviaqa": 64.2,
        "samsum": 37.16,
        "passage_retrieval_en": 5.5,
        "lcc": 35.53,
        "repobench-p": 39.95,
        "passage_count": 0.0
    },
    "task_average_result": {
        "single_doc_qa": 24.37,
        "multi_doc_qa": 20.93,
        "summarization": 14.58,
        "few_shots": 55.12,
        "synthetic": 2.75,
        "code": 37.74
    },
    "LB_average_result": 28.4
}
```