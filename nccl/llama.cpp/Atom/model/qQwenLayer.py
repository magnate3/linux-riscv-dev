import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Attention, Qwen3MLP
from quant import Quantizer, fake_quantize_quarter_E5M2, fake_quantize_quarter_E4M3, quantize_tensor, quantize_tensor_channel_group
from qLinearLayer import QLinearLayer

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,  # Qwen3特有的滑动窗口参数
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # 处理滑动窗口注意力（如果启用）
    if sliding_window is not None:
        # 创建滑动窗口掩码
        seq_len = query.shape[2]
        if seq_len > sliding_window:
            # 对于滑动窗口注意力，每个token只能看到前面的sliding_window个token
            sliding_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device), diagonal=0)
            sliding_mask = sliding_mask.triu(diagonal=-sliding_window + 1)
            sliding_mask = (1.0 - sliding_mask) * torch.finfo(query.dtype).min
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            if attention_mask is not None:
                # 结合因果掩码和滑动窗口掩码
                attn_weights = attn_weights + torch.max(attention_mask, sliding_mask)
            else:
                attn_weights = attn_weights + sliding_mask
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask
    elif attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Attention, Qwen3MLP
from quant import Quantizer
from qLinearLayer import QLinearLayer

class QQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: Qwen3DecoderLayer,
        args
    ):
        super().__init__()
        self.args = args
        self.hidden_size = originalLayer.hidden_size
        # 保留Qwen3特有的注意力类型配置
        self.attention_type = getattr(originalLayer, 'attention_type', 'full_attention')
        
        # 量化版本的注意力机制
        self.self_attn = QQwen3Attention(
            originalLayer.self_attn,
            args
        )
        # 量化版本的MLP
        self.mlp = QQwen3MLP(
            originalLayer.mlp,
            args
        )
        # 量化版本的层归一化
        self.input_layernorm = QQwen3RMSNorm(
            originalLayer.input_layernorm, 
            args
        )
        self.post_attention_layernorm = QQwen3RMSNorm(
            originalLayer.post_attention_layernorm, 
            args
        )

    def to(self, *args, **kwargs):
        super(QQwen3DecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Qwen3解码器层的前向传播
        保持与原始Qwen3DecoderLayer相同的结构，但插入量化操作
        """
        residual = hidden_states

        # 输入层归一化 + 量化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力机制 + 量化
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            attention_type=self.attention_type,  # 传递Qwen3特有的注意力类型
            **kwargs,
        )
        
        # 残差连接
        hidden_states = residual + hidden_states

        # 全连接层部分
        residual = hidden_states
        # 后注意力层归一化 + 量化
        hidden_states = self.post_attention_layernorm(hidden_states)
        # MLP + 量化
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states
        
        return hidden_states

class QQwen3RMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: Qwen3RMSNorm,
        args
    ):
        super().__init__()
        self.originalNorm = originalNorm
        self.act_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)
        self.args = args

    @torch.no_grad()
    def forward(self, hidden_states):
        result = self.originalNorm(hidden_states)
        if self.reorder_index is not None:
            assert result.shape[result.dim()-1] == self.reorder_index.shape[0]
            result = torch.index_select(result, result.dim()-1, self.reorder_index)

        if self.args.abits < 16:
            result = self.act_quant(result)

        return result
    
    def to(self, *args, **kwargs):
        super(QQwen3RMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self
class QQwen3Attention(nn.Module):
    def __init__(
        self, 
        originalAttn: Qwen3Attention,
        args
    ):
        super().__init__()
        self.abits = args.abits
        self.q_kv_cache = args.kv_cache
        self.config = originalAttn.config
        self.layer_idx = originalAttn.layer_idx
        self.head_dim = originalAttn.head_dim
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.scaling = originalAttn.scaling
        self.attention_dropout = originalAttn.attention_dropout
        self.is_causal = originalAttn.is_causal
        self.sliding_window = originalAttn.sliding_window  # Qwen3特有的滑动窗口
        
        # 量化线性层
        self.q_proj = QLinearLayer(originalAttn.q_proj, args)
        self.k_proj = QLinearLayer(originalAttn.k_proj, args)
        self.v_proj = QLinearLayer(originalAttn.v_proj, args)
        self.o_proj = QLinearLayer(originalAttn.o_proj, args)
        
        # Qwen3特有的查询和键归一化层 + 量化
        self.q_norm = originalAttn.q_norm
        self.k_norm = originalAttn.k_norm
        
        # 激活量化器
        self.act_quant = Quantizer(args=args)
        self.v_quant = Quantizer(args=args)
        self.k_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)

    def to(self, *args, **kwargs):
        super(QQwen3Attention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.q_norm = self.q_norm.to(*args, **kwargs)
        self.k_norm = self.k_norm.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        self.v_quant = self.v_quant.to(*args, **kwargs)
        self.k_quant = self.k_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_type: str = 'full_attention',
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3特有的：在投影后应用RMSNorm
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # 旋转位置编码
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV缓存量化
        if self.q_kv_cache:
            key_states = self.k_quant(key_states)

        # 更新过去键值对
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # KV缓存量化
        if self.q_kv_cache:
            value_states = self.v_quant(value_states)

        # 注意力计算（支持滑动窗口）
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window if attention_type == 'sliding_attention' else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # 重排序
        if self.reorder_index is not None:
            attn_output = torch.index_select(attn_output, 2, self.reorder_index)

        # 注意力输出量化
        attn_output = self.act_quant(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class QQwen3MLP(nn.Module):
    def __init__(
        self,
        originalMLP: Qwen3MLP,
        args
    ):
        super().__init__()
        # Qwen3的MLP没有偏置
        self.gate_proj = QLinearLayer(originalMLP.gate_proj, args)
        self.down_proj = QLinearLayer(originalMLP.down_proj, args)
        self.up_proj = QLinearLayer(originalMLP.up_proj, args)
        self.act_fn = originalMLP.act_fn
        self.act_quant = Quantizer(args=args)

    def to(self, *args, **kwargs):
        super(QQwen3MLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # Qwen3使用SwiGLU激活函数
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # 激活量化
        tmpResult = self.act_quant(tmpResult)
        return self.down_proj(tmpResult)