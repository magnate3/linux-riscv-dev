import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
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
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class QLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: LlamaDecoderLayer,
        args
    ):
        super().__init__()
        self.args = args
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QLlamaAttention(
            originalLayer.self_attn,
            args
        )
        self.mlp = QLlamaMLP(
            originalLayer.mlp,
            args
        )
        self.input_layernorm = QLlamaRMSNorm(
            originalLayer.input_layernorm, 
            args
        )
        self.post_attention_layernorm = QLlamaRMSNorm(
            originalLayer.post_attention_layernorm, 
            args
        )

    def to(self, *args, **kwargs):
        super(QLlamaDecoderLayer, self).to(*args, **kwargs)
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class QLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: LlamaRMSNorm,
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
        super(QLlamaRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

class QLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        originalAttn: LlamaAttention,
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
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = getattr(self.config, "rope_theta", 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = QLinearLayer(
            originalAttn.q_proj,
            args
        )
        self.k_proj = QLinearLayer(
            originalAttn.k_proj,
            args
        )
        self.v_proj = QLinearLayer(
            originalAttn.v_proj,
            args
        )
        self.o_proj = QLinearLayer(
            originalAttn.o_proj,
            args
        )
        self.act_quant = Quantizer(args=args)
        self.v_quant = Quantizer(args=args)
        self.k_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)

    def to(self, *args, **kwargs):
        super(QLlamaAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
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
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Fake quantize the key_states
        if self.q_kv_cache:
            key_states = self.k_quant(key_states)

        if past_key_values is not None:
            # For compatibility with new Cache interface
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Fake quantize the value_states
        if self.q_kv_cache:
            value_states = self.v_quant(value_states)

        # Use eager attention implementation
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # Reorder the BMM output to feed into o.proj
        if self.reorder_index is not None:
            attn_output = torch.index_select(attn_output, 2, self.reorder_index)

        # Quantize the attention output
        attn_output = self.act_quant(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    

class QLlamaMLP(nn.Module):
    def __init__(
        self,
        originalMLP: LlamaMLP,
        args
    ):
        super().__init__()
        self.gate_proj = QLinearLayer(
            originalMLP.gate_proj,
            args
        )
        self.down_proj = QLinearLayer(
            originalMLP.down_proj,
            args
        )
        self.up_proj = QLinearLayer(
            originalMLP.up_proj,
            args
        )
        self.act_fn = originalMLP.act_fn
        self.act_quant = Quantizer(args=args)

    def to(self, *args, **kwargs):
        super(QLlamaMLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Quantize the activations and feed into down_proj
        tmpResult = self.act_quant(tmpResult)
        return self.down_proj(tmpResult)