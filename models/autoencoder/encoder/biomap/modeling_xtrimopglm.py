""" PyTorch xTrimoPGLM model. """

import math
import copy
import warnings
import re
import sys
import os
import pathlib
import time
import random
import numpy as np
from tqdm.auto import tqdm

import torch
# D1 compatibility deviation from pinned Zaixi/RNAGenesis d8a4213:
# deepspeed is imported only by get_checkpoint_fn(), which runs when
# gradient checkpointing is enabled during training. The approved frozen
# eval path never calls it. Do not require DeepSpeed for the non-quantized
# forward. See README.md in this directory.
try:
    import deepspeed
except ImportError:
    deepspeed = None
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from copy import deepcopy
from collections import namedtuple
import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MaskedLMOutput,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers import PreTrainedModel
from transformers.utils import logging

from .configuration_xtrimopglm import xTrimoPGLMConfig
# D1 compatibility deviation from pinned Zaixi/RNAGenesis d8a4213:
# quantization.py is absent from the pinned revision. quantize() is reached
# only from xTrimoPGLMForMaskedLM when quantization_bit is non-zero. The
# pinned config sets quantization_bit = 0 and D1 constructs xTrimoPGLMModel,
# so this import is unused. No quantization behaviour is implemented.
try:
    from .quantization import quantize
except ImportError:
    def quantize(*args, **kwargs):
        raise RuntimeError(
            "xTrimoPGLM quantization was requested, but quantization.py is not "
            "part of pinned revision d8a42130984cbf04f6a5e16a3aa0c0d6578036a8. "
            "The approved D1 path uses quantization_bit=0 and does not quantize."
        )

def get_checkpoint_fn():
    if deepspeed is not None and deepspeed.checkpointing.is_configured():
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint
    return checkpoint

# flags required to enable jit fusion kernels

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "BioMap/xtrimopglm-100b-int4"
_CONFIG_FOR_DOC = "xTrimoPGLMConfig"
DeepNormCoefficients = namedtuple("DeepNormCoefficients", ["alpha", "beta"])

def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


def get_deepnorm_coefficients(config: xTrimoPGLMConfig):
    """
        DeepNorm coefficients from : https://kexue.fm/archives/8978
    """
    num_layers = config.num_layers
    return DeepNormCoefficients(alpha=(2 * num_layers) ** 0.5, beta=(2 * num_layers) ** -0.5)

"""
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
"""

def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim)).to(precision)
        self.dim = dim
        self.base = base
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f'{prefix}inv_freq' in state_dict:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        else:
            self.inv_freq.copy_(1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)).to(self.precision))

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16 or self.precision == torch.half:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            elif self.precision == torch.half:
                cos_cached = cos_cached.half()
                sin_cached = sin_cached.half()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class ConvLayer(nn.Module):
    def __init__(self, hidden_size, kernels = [3, 5, 7], N=10):
        super().__init__()
        self.conv3 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=7, stride=1, padding=3)
        self.conv9 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=9, stride=1, padding=4)
        self.conv11 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=11, stride=1, padding=5)
        self.conv13 = nn.Conv1d(hidden_size*N, hidden_size, kernel_size=13, stride=1, padding=6)        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_tensor = x.transpose(1, 2)

        out3 = self.relu(self.conv3(x_tensor))
        out5 = self.relu(self.conv5(x_tensor))
        out7 = self.relu(self.conv7(x_tensor))
        out9 = self.relu(self.conv9(x_tensor))
        out11 = self.relu(self.conv11(x_tensor))
        out13 = self.relu(self.conv13(x_tensor))        

        x_tensor = x_tensor + out3 + out5 + out7 + out9 + out11 + out13
        x_tensor = x_tensor.transpose(1, 2).transpose(0, 1) # [sq, b, hiddden_size]
        return x_tensor


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

def assert_dim_check(tensor, ndim=None, shape=None):
    if ndim is not None:
        assert tensor.ndim == ndim, f"Exepct tensor.ndim={ndim}. gut got tensor.shape={tensor.shape}"
    if shape is not None:
        assert list(tensor.shape) == list(shape), f"Exepct tensor.shape={shape}. gut got tensor.shape={tensor.shape}"

def apply_rotary_pos_emb_index_torch(q, k, cos, sin, position_id):  # jitting fails with bf16
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: xTrimoPGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

        self.is_causal = config.is_causal
        self.use_pytorch_sdpa = config.use_pytorch_sdpa
    
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # query_layer, key_layer, value_layer: [seq_len, batch_size, num_heads, head_dim]
        pytorch_major_version = int(torch.__version__.split('.')[0])
        # assert pytorch_major_version >= 2, f"Expect PyTorch version > 2.0"
        if pytorch_major_version >= 2 and self.use_pytorch_sdpa:
            dropout_p = self.attention_dropout.p if self.training else 0
            # [seq_len, batch_size, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            # import pdb; pdb.set_trace();
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                # context_layer: [batch_size, num_heads, seq_len, head_dim]
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, is_causal=self.is_causal, dropout_p=dropout_p)
            else:
                if (attention_mask is not None) and (attention_mask.dtype == torch.bool):
                    attention_mask = attention_mask.logical_not() ## DO NOT inplace operation!!!!
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, dropout_p=dropout_p)
            # [batch_size, num_heads, seq_len, head_dim] -> [seq_len, batch_size, num_heads, head_dim]
            context_layer = context_layer.permute(2, 0, 1, 3)
            # [seq_len, batch_size, 2560]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if self.is_causal and attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: xTrimoPGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear, device=device, **_config_to_kwargs(config))
        
        self.rotary_embedding_2d = config.rotary_embedding_2d
        # dim, base=10000, precision=torch.half, learnable=False
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head // 2 if self.rotary_embedding_2d else self.hidden_size_per_attention_head, 
                                          base=10000, precision=config.torch_dtype, learnable=False)


    def forward(
            self, hidden_states, attention_mask, position_ids, kv_cache=None, use_cache=True
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if position_ids is not None: # [seq_len, 2, batch_size, 32, 2]
            
            if self.rotary_embedding_2d:
                q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1)) # 32
                k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
                # import pdb; pdb.set_trace();
                cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1) # 32
                position_ids, block_position_ids = \
                    position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                    position_ids[:, 1, :].transpose(0, 1).contiguous()
                q1, k1 = apply_rotary_pos_emb_index_torch(q1, k1, cos, sin, position_ids)
                q2, k2 = apply_rotary_pos_emb_index_torch(q2, k2, cos, sin, block_position_ids)
                query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
                key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
            else:
                # [b, sq] -> [sq, b]
                position_ids = position_ids.transpose(0, 1)
                cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
                query_layer, key_layer = apply_rotary_pos_emb_index_torch(query_layer, key_layer, cos, sin, position_ids)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            key_layer = key_layer.contiguous().view(key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            value_layer = value_layer.contiguous().view(value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask) # context_layer: [seq_len, batch_size, num_heads*head_dim]
        output = self.dense(context_layer)
        # =================
        # Output. [sq, b, h]
        # =================

        # output = context_layer @ self.dense.weight.T + self.dense.bias
        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: xTrimoPGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear
        self.moe = config.moe
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token # 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
           x = torch.chunk(x, 2, dim=-1)
           return x[0] * F.silu(x[1])

        def geglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return x[0] * F.gelu(x[1])

        if config.glu_activation == 'geglu':
            self.activation_func = geglu
        elif config.glu_activation == 'swiglu':
            self.activation_func = swiglu
        else:
            assert RuntimeError(f"Unsupported glu_activation: {config.glu_activation}")

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        if self.moe:
            assert self.num_experts > 1
            del self.dense_h_to_4h
            del self.dense_4h_to_h
            self.router = nn.Linear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                device=device,
                dtype=torch.float32
            )
            for i in range(0, self.num_experts):
                self.register_module(f"dense_h_to_4h_{i}", nn.Linear(
                    config.hidden_size,
                    config.ffn_hidden_size * 2,
                    bias=self.add_bias,
                    device=device,
                    **_config_to_kwargs(config)
                ))
                self.register_module(f"dense_4h_to_h_{i}", nn.Linear(
                    config.ffn_hidden_size,
                    config.hidden_size,
                    bias=self.add_bias,
                    device=device,
                    **_config_to_kwargs(config)
                ))

    def moe_forward(self, hidden_states, expert_idx):
        intermediate_parallel = getattr(self, f"dense_h_to_4h_{expert_idx}")(hidden_states) 
        intermediate_parallel = self.activation_func(intermediate_parallel) 
        output = getattr(self, f"dense_4h_to_h_{expert_idx}")(intermediate_parallel) 
        return output

    def forward(self, hidden_states):
        if self.moe:
            # import pdb; pdb.set_trace();
            s, b, n = hidden_states.shape
            dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
            route = self.router(hidden_states).to(dtype)

            weights, selected_experts = torch.topk(route, self.experts_per_token)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(hidden_states.dtype)
            output = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            for expert_idx in range(self.num_experts):
                batch_idx, nth_expert = torch.where(selected_experts == expert_idx)
                if nth_expert.shape[0] == 0:
                    continue
                cur_out = self.moe_forward(hidden_states[batch_idx], expert_idx)
                output[batch_idx] += weights[batch_idx, nth_expert, None] * cur_out
            output = output.reshape(s, b, n)
        else:
            # [s, b, 4hp]
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            # [s, b, h]
            output = self.dense_4h_to_h(intermediate_parallel)
        return output

class xTrimoPGLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: xTrimoPGLMConfig, layer_number, device=None):
        super(xTrimoPGLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = MixedFusedLayerNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config, device=device)

        self.deepnorm_coeff = get_deepnorm_coefficients(config) if config.deepnorm else None

    def forward(
            self, hidden_states, attention_mask, position_ids, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            position_ids, # [batch_size, 2, seq_len, 32, 2]
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        if self.deepnorm_coeff is not None: 
            layernorm_input = residual*self.deepnorm_coeff.alpha + layernorm_input
        else:
            layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        if self.deepnorm_coeff is not None: 
            output = residual*self.deepnorm_coeff.alpha + output
        else:
            #print(f"2 self.deepnorm_coeff is None")
            output = residual + output

        return output, kv_cache


class xTrimoPGLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: xTrimoPGLMConfig, device=None):
        super(xTrimoPGLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return xTrimoPGLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(
                config.hidden_size, eps=config.layernorm_epsilon, bias=True
            )

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, position_ids, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_ret = get_checkpoint_fn()(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)


        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class xTrimoPGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = xTrimoPGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["xTrimoPGLMBlock"]

    _quantized = False


    def get_masks(self, input_ids, past_key_values, padding_mask=None, is_causal=True):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        if is_causal:
            full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device, context_length=0):
        batch_size, seq_length = input_ids.shape
        if self.config.rotary_embedding_2d:
            if self.config.is_causal: # 100b model
                position_ids_1 = torch.zeros(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
                position_ids_2 = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
                position_ids   = torch.stack([position_ids_1, position_ids_2], axis=1) # [batch_size, 2, seq_len]       
            else:
                position_ids_1 = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
                position_ids_2 = torch.zeros(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
                position_ids   = torch.stack([position_ids_1, position_ids_2], axis=1) # [batch_size, 2, seq_len]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, 1, seq_len]
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, xTrimoPGLMTransformer):
            module.gradient_checkpointing = value

    
    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        std = self.config.initializer_range
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def quantize(self, weight_bit_width: int, empty_init=True, device=None):
        if self._quantized:
            print(f"Model has been quantized...")
            return
        self.transformer.encoder = quantize(self.transformer.encoder, weight_bit_width, empty_init, device)
        self._quantized = True
        return self

class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: xTrimoPGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection


    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        
        embeddings = embeddings.contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings

class xTrimoPGLMModel(xTrimoPGLMPreTrainedModel):
    def __init__(self, config: xTrimoPGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.conv_layer = ConvLayer(config.hidden_size, "3,5,7,9,11,13", 1)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        # self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, base=10000, precision=config.torch_dtype, learnable=False)
        self.encoder = init_method(xTrimoPGLMTransformer, config, **init_kwargs)
        
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=True,
                                        dtype=config.torch_dtype, **init_kwargs)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None, # position_ids: [batch_size, 2, seq_len]
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.config.is_causal:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        inputs_embeds = self.conv_layer(inputs_embeds)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, position_ids=position_ids,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class xTrimoPGLMForMaskedLM(xTrimoPGLMPreTrainedModel):
    def __init__(self, config: xTrimoPGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = xTrimoPGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        if self.config.quantization_bit:
            print(f"Begin Quantization to {self.config.quantization_bit} bit")
            self.quantize(self.config.quantization_bit, empty_init=True, device=device)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = None,
            return_last_hidden_state: Optional[bool] = None
    ):
        if self.config.is_causal:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)

        full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask, is_causal=self.config.is_causal)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids, # position_ids: [batch_size, 2, seq_len]
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        masked_lm_loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100) # -100 for padding token.
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(
            loss = masked_lm_loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.last_hidden_state if return_last_hidden_state else transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
