
#  Mistral-7B-v0.1
```
modelscope download --model 'AI-ModelScope/Mistral-7B-v0.1' --cache_dir '/workspace/models/Mistral-7B-v0.1'
```

# calculate_shannon_entropy

```
import numpy as np
from scipy.stats import entropy

def calculate_shannon_entropy(tensor_data, bins=256, val_range=(-4.0, 4.0)):
    # Flatten tensor
    flat_data = tensor_data.numpy().flatten()
    
    # Calculate histogram
    counts, _ = np.histogram(flat_data, bins=bins, range=val_range)
    
    # Check for empty data/all zeros
    if np.sum(counts) == 0:
        return 0.0
        
    # Calculate probabilities
    probs = counts / np.sum(counts)
    
    # Return entropy in bits
    return entropy(probs, base=2)

```
对于两组取值范围不同的数据，采用固定的 val_range=(-4.0, 4.0) 会导致截断误差和分辨率失真，从而使计算出的香农熵（Shannon Entropy）无法真实反映其内部的复杂性。具体影响可以分为以下三种情况：
1. 数据超出固定范围（数据范围 > 4.0 或 < -4.0）信息丢失（截断）：超出 [-4, 4] 范围的所有数据都会被 np.histogram 直接丢弃，不计入频数。熵值偏低：由于大量边界外的数据被忽略，计算出的概率分布会失真，导致计算出的熵值比实际值明显偏低。    
2. 数据远小于固定范围（例如实际范围在 0 到 0.1 之间）分辨率骤降（稀疏化）：256个 bin 是均匀分配在 [-4, 4] 区间内的（每个 bin 宽度约为 0.031）。如果数据高度集中在 0 到 0.1，它们只会落入其中 3~4 个 bin 中，其余 250 多个 bin 的频数全为 0。无法区分细节：这相当于对数据进行了过度粗糙的量化。两组原本分布完全不同的微观数据，可能会因为落入相同的几个 bin 而算出几乎相同的低熵值。    
3. 数据本身就在固定范围内（但两组数据分布不同）具有可比性（唯一优点）：如果两组数据都在 [-4, 4] 内（例如一组在 [-1, 1]，另一组在 [-3, 3]），使用相同的坐标系和 bin 宽度，可以让你公平地比较两组数据的绝对混乱度。    