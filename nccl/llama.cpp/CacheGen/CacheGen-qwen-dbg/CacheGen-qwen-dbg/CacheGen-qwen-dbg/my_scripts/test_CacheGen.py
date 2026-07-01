import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer

# 1. 准备配置与元数据
config = LMCacheEngineConfig.from_defaults(chunk_size=256, remote_serde="cachegen")
# 注意：需根据实际情况填写 metadata，此处仅为示例
metadata = LMCacheEngineMetadata(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    world_size=1, worker_id=0, fmt="vllm",
    kv_dtype=torch.float16, kv_shape=(32, 2, 256, 8, 128)
)

# 2. 初始化序列化器
serializer = CacheGenSerializer(config, metadata)

# 3. 创建模拟 KV 数据
if torch.cuda.is_available():
    # 模拟一个 256 tokens 的 KV Cache
    kv_tensor = torch.rand((32, 2, 256, 8, 128), device="cuda", dtype=torch.float16)

    # 4. 执行压缩
    compressed_bytes = serializer.to_bytes(kv_tensor)

    original_size = kv_tensor.nelement() * 2
    compressed_size = len(compressed_bytes)
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")
else:
    print("CacheGen requires CUDA to run.")
