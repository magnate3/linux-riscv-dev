

[benchmarking/setup/patch_vllm_for_lmcache_cacheblend.py](https://github.com/RaviTejGuntuku/kv_cache_optimization/blob/daff2c492dfc3c34c263671ba0f442e3f18baeea/benchmarking/setup/patch_vllm_for_lmcache_cacheblend.py)    

[demo-rag-blending](https://github.com/LMCache/LMCache-Examples/tree/19b31ac24b37bbbd260cb701d07a916d4b1f7ed8/demo-rag-blending)      

[LMCache/benchmarks/rag](https://github.com/Celeste025/pic_project/tree/d2c151dbe86bc308fce75d055612b68bc584a8d2/LMCache/benchmarks/rag)     

[LMCache/examples/blend_kv_v1/blend.py](https://github.com/Celeste025/pic_project/blob/d2c151dbe86bc308fce75d055612b68bc584a8d2/LMCache/examples/blend_kv_v1/blend.py#L157)    


(CacheBlend/example)[https://github.com/YaoJiayi/CacheBlend/blob/main/README.md]

# Example run


## Run Musique dataset

Compare LLM inference with CacheBlend and normal prefill
```
python example/blend_musique.py
```
To run datasets other than musique, please replace `musique` with `samsum` or `wikimqa` in the above command.

## Run LLM inference with CacheBlend
```
python example/blend.py
```
 blend.py

```
export VLLM_VERSION=v0
export VLLM_USE_V1=0
# 1. 核心开关：强制开启 CacheBlend 功能
export LMCACHE_ENABLE_BLENDING=True

# 2. 核心开关：启用层级（Layer-wise）张量控制，Blending 模式下必须为 True
export LMCACHE_USE_LAYERWISE=True

# 3. 指定块间的多段文本切割占位符（例如在 RAG 中用于拼接不同 Chunk 文档）
export LMCACHE_BLEND_SPECIAL_STR=" # # "

# 4. 指定在高重要度层（通常为第 1 层）计算 KV 偏离度并选择性重算 Token 的比例（官方推荐 15%）
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15

# 5. 【极其关键】指定 vLLM 注意力后端为 FlashInfer，Blending 算子强依赖它来进行块拼接
export VLLM_ATTENTION_BACKEND=FLASHINFER

```

```
export LMCACHE_LOG_LEVEL=ERROR
```

```
python3 blend.py  --model /models/qwen2.5-0.5b   --use-disk
```
```
EngineCore pid=16296) WARNING 07-23 08:59:31 [jit_monitor.py:106] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 19.65it/s, est. speed input: 39366.77 toks/s, output: 19.69 toks/s]
--------------------------------------------------
Generated text: 'Nice'
Generation took 0.06 seconds, warmup request done.
--------------------------------------------------
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2383.13it/s]
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.52it/s, est. speed input: 45373.69 toks/s, output: 4.53 toks/s]
--------------------------------------------------
Generated text: 'You'
Generation took 0.22 seconds, first request done.
--------------------------------------------------
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2554.39it/s]
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.74it/s, est. speed input: 57643.55 toks/s, output: 5.75 toks/s]
--------------------------------------------------
Generated text: 'You'
Generation took 0.17 seconds, second (warming up blend code path) request done.
--------------------------------------------------
Rendering prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2531.26it/s]
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.59it/s, est. speed input: 96315.46 toks/s, output: 9.60 toks/s]
--------------------------------------------------
Generated text: 'You'
Generation took 0.11 seconds, third request done.
--------------------------------------------------
```

+ 磁盘文件

```
du -sh local_disk/
334M    local_disk/
```
