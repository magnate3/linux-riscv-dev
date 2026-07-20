# RoE: Roster of Experts — 基于 vLLM 的复现与实现

本项目复现并工程化了 Apple 机器学习团队提出的 Roster of Experts (RoE) 推理框架。在原始 [vLLM](https://github.com/vllm-project/vllm) 推理引擎基础上，增加了 RoE 的门控机制、Gumbel 噪声扰动、多专家并行执行以及 log-sum-exp 聚合机制，从而在不改变模型结构的前提下提升推理鲁棒性与多样性。

---

## 项目简介

RoE（Roster of Experts）针对混合专家模型（Mixture-of-Experts, MoE）提出高效推理方法：
- 在每个解码步（decoding step）同时执行 K 路不同的专家路由；
- 在门控 logits 上加入 Gumbel 噪声（温度 τ）引入多样性；
- 使用 log-sum-exp 对各路输出进行稳定聚合，得到更鲁棒的 token 分布。

---

## 主要特性

- 动态启用/禁用 RoE（与 vLLM 原生行为向后兼容）
- K 路专家采样机制（多样路由并行）
- 可控的 Gumbel 噪声扰动（τ 参数）
- Log-Sum-Exp logits 聚合
- 专家路由多样性调试输出（如 Jaccard Overlap）

---



---

## 环境搭建

### 1) 克隆仓库与子模块

```bash
git clone --recurse-submodules git@github.com/ShengxuanQiu/RoE.git
cd RoE
# 若子模块未自动初始化：
git submodule update --init --recursive
```

### 2) 创建并激活 Python 环境

```bash
conda create -n roe python=3.12 -y
conda activate roe
```

### 3) 本地编译安装 vLLM（可编辑模式）

```bash
cd vllm
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install --editable .
cd ..
```

---

## 运行示例

以下示例默认读取数据集文件 `/RoE/gsm8K.jsonl`，并将结果写入 `/RoE/outputs.jsonl`。

- 关闭 RoE（baseline）：
```bash
CUDA_VISIBLE_DEVICES=1,2 python roe.py \
  --enable-roe 0 \
  --roe-num-samples 1 \
  --roe-tau 0.0
```

- 启用 RoE（K=8, τ=0.3）：
```bash
CUDA_VISIBLE_DEVICES=1,2 python roe.py \
  --enable-roe \
  --roe-num-samples 8 \
  --roe-tau 0.3
```

说明：
- 脚本会读取数据集中全部问题，按批处理（脚本内常量 BATCH_SIZE 控制，默认 1）。
- 每条样本的预测、标准答案与正误会写入 outputs.jsonl（JSONL 一行一条）。
- 所有问题处理完成后，会在文件末尾追加 summary（total、correct、accuracy）。

---

## 参数说明

| 参数 | 含义 | 默认值 | 示例 |
|---|---|---|---|
| --model | 模型路径或名称 | 脚本内默认 | --model /path/to/model |
| --questions-file | JSONL 数据集文件 | /data/home/shengxuanqiu/RoE/gsm8K.jsonl | --questions-file ./gsm8K.jsonl |
| --enable-roe | 是否启用 RoE（布尔） | False | --enable-roe 或 --enable-roe 0 |
| --roe-num-samples | 每步解码的专家采样次数 K | 1 | --roe-num-samples 8 |
| --roe-tau | Gumbel 噪声温度 τ（可多值） | 0.0 | --roe-tau 0.3 |
| --tensor-parallel-size | 张量并行度 | 1 | --tensor-parallel-size 2 |
| --dtype | 模型精度 | auto | --dtype float16 |
| --max-new-tokens | 生成最大新 token 数 | 256 | --max-new-tokens 256 |
| --temperature | 采样温度 | 0.0 | --temperature 0.7 |
| --output | 输出 JSONL 文件路径 | /data/home/shengxuanqiu/RoE/outputs.jsonl | --output ./outputs.jsonl |
| --log-prompts | 打印部分题目预览 | 关闭 | --log-prompts |

提示：
- 本仓库当前脚本使用自定义的布尔解析（str2bool），关闭可用“--enable-roe 0”，开启可用“--enable-roe”。

---

## 结果文件说明

- 每条样本输出字段包含：qid、question、response、pred_answer（从模型回复中提取的最后一个数字）、gold_answer（#### 后的标准答案）、correct 等。
- 文件末尾 summary 示例：
```json
{"summary": {"total": 100, "correct": 85, "accuracy": 0.85}}
```

---

## 实现细节

RoE 相关核心改动（位于 vLLM 子模块）：
- 新增 RoeConfig 管理运行参数
- 在专家门控处注入 Gumbel 噪声扰动
- 解码步内执行 K 路并行前向与共享 KV Cache
- 使用 log-sum-exp 聚合多路 logits，并在聚合后应用 logits processors（避免重复过滤）

---

## 参考论文

- Apple Machine Learning Research, “MoEs Are Stronger Than You Think: Hyper-Parallel Inference Scaling with RoE”

以上工作在 vLLM 框架基础上进行复现与集成。