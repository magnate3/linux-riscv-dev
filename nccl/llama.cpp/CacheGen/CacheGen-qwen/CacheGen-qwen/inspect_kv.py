import os
import torch

def inspect_kv_cache_raw_data(file_path="./kv_output/raw_kv_0.pt"):
    """
    读取并解析 CacheGen 第 1 阶段收集的原始 KV Cache 原材料信息
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：未找到文件 {file_path}，请确保第 1 阶段 main.py 已经成功运行。")
        return

    # 2. 计算精确的磁盘占用字节数
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print("=" * 60)
    print(f"📦 原始 KV Cache 磁盘文件信息")
    print("=" * 60)
    print(f"📂 文件路径: {file_path}")
    print(f"💾 精确磁盘占用: {file_size_bytes:,} 字节 (Bytes)")
    print(f"📊 换算体积: {file_size_mb:.2f} MB")
    print("-" * 60)

    # 3. 加载并解析 PyTorch 张量结构
    try:
        # 加上 weights_only=True 保证新版 PyTorch 的加载安全性
        raw_kv = torch.load(file_path, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return

    # 4. 分析 LMCache/Transformers 的 KV 组织格式
    # 不同的 transformers 版本返回的 past_key_values 可能是元组、列表或特定的 Cache 类
    num_layers = 0
    if isinstance(raw_kv, (tuple, list)):
        num_layers = len(raw_kv)
        first_layer = raw_kv[0]
        
        # 检查这一层的内容 (通常是包含两个张量的元组: (key, value))
        if isinstance(first_layer, (tuple, list)) and len(first_layer) >= 2:
            key_tensor, value_tensor = first_layer[0], first_layer[1]
            
            print(f"🧱 神经网络总层数 (Num Layers): {num_layers}")
            print(f"🎨 数据类型 (Dtype):")
            print(f"   - Key 缓存: {key_tensor.dtype}")
            print(f"   - Value 缓存: {value_tensor.dtype}")
            print(f"📐 核心张量维度 (Shape):")
            print(f"   - Key 张量结构 [Batch, Heads, Seq_Len, Dim]: {list(key_tensor.shape)}")
            print(f"   - Value 张量结构 [Batch, Heads, Seq_Len, Dim]: {list(value_tensor.shape)}")
            print(f"🔤 捕获的上下文长度 (Sequence Length): {key_tensor.shape[2]} Token")
        else:
            print(f"⚠️ 无法识别的层内数据结构类型: {type(first_layer)}")
            
    elif hasattr(raw_kv, 'key_cache') and hasattr(raw_kv, 'value_cache'):
        # 兼容新版 transformers 返回的 DynamicCache 对象
        num_layers = len(raw_kv.key_cache)
        key_tensor = raw_kv.key_cache[0]
        value_tensor = raw_kv.value_cache[0]
        print(f"🧱 检测到新版 DynamicCache 结构，总层数: {num_layers}")
        print(f"📐 第一层 Key Shape: {list(key_tensor.shape)}, Value Shape: {list(value_tensor.shape)}")
    else:
        print(f"⚠️ 无法识别的全局数据结构类型: {type(raw_kv)}")
        
    print("=" * 60)

if __name__ == "__main__":
    inspect_kv_cache_raw_data("./kv_output/raw_kv_0.pt")

