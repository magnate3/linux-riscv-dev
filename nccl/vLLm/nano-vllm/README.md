
# cpu

```
[root@centos7 nano-vllm-build]# docker run -it  -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name nano-vllm nano-vllm-openai-cpu:latest-arm64 bash
root@6125fede8db4:/vllm-workspace# cd /workspace/nano-vllm-cpu
root@6125fede8db4:/workspace/nano-vllm-cpu# ls
LICENSE  README.md  bench.py  example.py  img.png  nanovllm  requirements.txt
root@6125fede8db4:/workspace/nano-vllm-cpu# python3 example.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [35:42<00:00, 714.00s/it, Prefill=0tok/s, Decode=0tok/s]


Prompt: '<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n'
len 191 Completion: '<think>\n嗯，用户问我是谁。我是一个AI助手，但需要保持诚实和友善。我不会提供任何关于我的个人信息或身份的虚假回答。我专注于帮助用户解决问题和获取信息。如果您有其他问题或需要帮助，请随时告诉我！\n</think>\n\n我是你的虚拟助手，专注于帮助你解决问题和获取信息。我不会提供任何关于我的个人信息或身份的虚假回答。如果你有任何问题或需要帮助，请随时告诉我！<|im_end|>'


Prompt: '<|im_start|>user\n中国是<|im_end|>\n<|im_start|>assistant\n'
len 228 Completion: '<think>\n好的，用户问“中国是”，我需要回答。首先，我应该确认用户的问题意图，可能是在测试我的回答能力，或者想了解中国的基本情况。用户可能对中国的国家身份、领土、人口等信息感兴趣。\n\n接下来，我要确保回答准确且简洁。中国是一个中华人民共和国，国家主席是习近平，人民代表大会制度是其基本政治制度，国家主席由全国人民推选产生。同时，中国有多个少数民族，如壮族、蒙古族等，这些都与国家的多民族特征相关。\n\n还要注意用户可能的误解，比如将“国家”与“国家统一'


Prompt: '<|im_start|>user\n讲一个笑话，题材任意，200字<|im_end|>\n<|im_start|>assistant\n'
len 221 Completion: '<think>\n好的，用户让我讲一个笑话。首先，我需要确定笑话的主题。用户可能想要轻松有趣的内容，所以得选一个常见的元素，比如日常生活中的小误会或者反差。\n\n接下来，我得考虑结构。通常笑话会有开头、转折和结尾。比如，可以讲一个关于时间或天气的笑话，这样容易引起共鸣。比如，时间过得快或者天气预报的错误，这样比较贴近生活。\n\n然后，我需要确保笑话在200字左右，不要太长。可能需要加入一些细节，比如对话或者角色，让笑话更生动。比如，一个老奶奶和'
root@6125fede8db4:/workspace/nano-vllm-cpu# 
```

#  cpu

```
[root@centos7 nano-vllm-build]# docker run -it  -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name nano-vllm nano-vllm-openai-cpu:latest-arm64 bash
```

+ uninstall  nano-vllm
```
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# pip uninstall  nano-vllm   
Found existing installation: nano-vllm 0.2.0
Uninstalling nano-vllm-0.2.0:
  Would remove:
    /opt/venv/lib/python3.12/site-packages/nano_vllm-0.2.0.dist-info/*
    /opt/venv/lib/python3.12/site-packages/nanovllm/*
Proceed (Y/n)? y
  Successfully uninstalled nano-vllm-0.2.0
```

```
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# cat example.py 
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/workspace/qwen/models/Qwen3-0.6B")
    #path = os.path.expanduser("/workspace/qwen/models/qwen2-7b-instruct")
    #path = os.path.expanduser("D:\models\Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path)
    # max_tokens太差会报错
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "你是谁",
        "中国是",
        "讲一个笑话，题材任意，200字",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"len {len(output['text'])} Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
```

```
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# pwd
/workspace/nano-vllm-cpu
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# export export PYTHONPATH=/workspace/nano-vllm-cpu/nanovllm 
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# export PYTHONPATH=/workspace/nano-vllm-cpu/nanovllm
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# python3 example.py
```

```
root@c8e3b759fb0d:/workspace/nano-vllm-cpu# python3 example.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [31:46<00:00, 635.37s/it, Prefill=0tok/s, Decode=0tok/s]


Prompt: '<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n'
len 228 Completion: '<think>\n嗯，用户问我是谁。我是一个AI助手，但需要保持身份的正确性。首先，我应该确认自己的身份，比如以“我是一个AI助手”来回应。同时，要让用户感到被尊重，他们可以继续提问，而不会感到尴尬。\n\n在回答时，可以简短明了，避免冗长。使用友好而专业的语气，让用户觉得我是一个可靠的服务伙伴。另外，可以询问用户是否有需要帮助的地方，这样可以促进更多的互动。\n\n需要注意的是，回答要简洁，避免使用复杂句子结构，让用户容易理解和接受。同时，保持礼貌，让用户感到'


Prompt: '<|im_start|>user\n中国是<|im_end|>\n<|im_start|>assistant\n'
len 235 Completion: '<think>\n好的，用户问“中国是”，我需要先确认他们的问题意图。用户可能是在询问中国的基本信息，或者想了解其他关于中国的内容。我应该保持回答的简洁和准确，同时提供一些有用的信息。\n\n首先，我需要确认用户是否想知道中国的主要特征，比如人口、经济、地理等。可能需要涵盖这些方面，让回答全面。同时，要注意用词准确，避免任何可能引起误解的地方。\n\n另外，用户可能是在进行某种测试，或者有其他目的。我应该保持回答友好，避免过于正式，让用户感到被理解和尊重。同时，检查是否有任何'


Prompt: '<|im_start|>user\n讲一个笑话，题材任意，200字<|im_end|>\n<|im_start|>assistant\n'
len 228 Completion: '<think>\n好的，用户让我讲一个笑话，题材任意，200字。首先，我需要确定一个合适的主题。笑话通常需要一个有趣的情节，所以可能选择一些常见但容易引发笑点的元素。\n\n比如，常见的流行梗，比如“上班迟到”或者“办公室里的搞笑事件”。或者更幽默的设定，比如时间旅行或者奇怪的日常。考虑到用户可能希望内容轻松，可能选择一个日常办公室的场景会比较合适。\n\n接下来，我需要构建一个场景。比如，同事在办公室里因为工作失误导致尴尬，或者遇到奇怪的办公室现象。比如，有人'
```
# gpu

+ apt-get -y install cuda   
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

```
apt-get install -y gnupg
```


```
root@ubuntu:/workspace/nano-vllm# apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
Warning: apt-key is deprecated. Manage keyring files in trusted.gpg.d instead (see apt-key(8)).
OK
```


+ 通过runfile手动安装cuda (apt-get -y install cuda) 
```
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux_sbsa.run
```


```
root@ubuntu:~/pytorch/nano-vllm$ sudo docker build -t nano-vllm-gpu -f Dockerfil-runfile . 
[+] Building 161.7s (9/9) FINISHED                                                                                                                                                docker:default
 => [internal] load build definition from Dockerfil-runfile                                                                                                                                 0.0s
 => => transferring dockerfile: 1.05kB                                                                                                                                                      0.0s
 => [internal] load metadata for docker.io/library/nano-vllm:latest                                                                                                                         0.0s
 => [internal] load .dockerignore                                                                                                                                                           0.0s
 => => transferring context: 2B                                                                                                                                                             0.0s
 => [internal] load build context                                                                                                                                                           0.0s
 => => transferring context: 61B                                                                                                                                                            0.0s
 => [1/4] FROM docker.io/library/nano-vllm:latest                                                                                                                                           0.0s
 => CACHED [2/4] COPY    cuda_11.6.0_510.39.01_linux_sbsa.run  /tmp/cuda_11.6.0_510.39.01_linux_sbsa.run                                                                                    0.0s
 => CACHED [3/4] RUN apt-get update && apt-get install -y build-essential                                                                                                                   0.0s
 => [4/4] RUN   sh /tmp/cuda_11.6.0_510.39.01_linux_sbsa.run --silent --toolkit --override                                                                                                140.3s
 => exporting to image                                                                                                                                                                     21.3s
 => => exporting layers                                                                                                                                                                    21.3s
 => => writing image sha256:2267f7b4efbbc9a8bc8cb9b0834aad570b24f64960b948440c26338378ebf169                                                                                                0.0s
 => => naming to docker.io/library/nano-vllm-gpu 
```