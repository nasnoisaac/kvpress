from dataclasses import dataclass

import torch
from torch import nn
import numpy as np

from kvpress.presses.scorer_press import ScorerPress
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from transformers import  QuantizedCache

import math
from torch.nn import functional as F

@dataclass
class AdapPress(ScorerPress):

    compression_ratio: float = 0.0
    layer_gini_scores: list = None
    num_layers: int = None
    window_size: int = 64

    def __post_init__(self):
        self.layer_gini_scores = []
        self.num_layers = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return -keys.norm(dim=-1)

    def prefill_gini(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ):
        # assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        if attentions is None:
            attentions = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
        
        # Initialize layer_gini_scores if first layer
        if module.layer_idx == 0:
            self.num_layers = module.config.num_hidden_layers
            self.layer_gini_scores = [0.0] * self.num_layers

        # Calculate and update gini score for current layer
        if not all(score != 0.0 for score in self.layer_gini_scores):
            gini = self.calculate_gini(attentions)
            self.layer_gini_scores[module.layer_idx] = gini


    def calculate_gini(self, attentions):
        """Calculate Gini coefficient (measure of inequality/sparsity) directly on GPU"""
        # Average across batch and heads
        attn = attentions.mean(dim=(0, 1))
        
        # Sort the attention values
        sorted_attn, _ = torch.sort(attn.flatten())
    
        # Calculate Gini coefficient
        n = sorted_attn.shape[0]
        index = torch.arange(1, n + 1, device=sorted_attn.device)
        numerator = torch.sum((2 * index - n - 1) * sorted_attn)
        denominator = n * torch.sum(sorted_attn)
        
        # Clear intermediate tensors
        del sorted_attn, index
        torch.cuda.empty_cache()
        
        return (numerator / denominator).item()


        

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]

        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights



    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self.prefill_gini(module, hidden_states, keys, values, attentions, kwargs)

        if module.layer_idx == module.config.num_hidden_layers - 1:
            self.post_prefilling(module, hidden_states, kwargs)

        return keys, values

 


    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache while ensuring:
            - compression is only applied only during the pre-filling phase
            - KV cache quantization is handled correctly

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]
        q_len = hidden_states.shape[1]


        if kwargs["cache_position"][-1] > q_len:
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.empty_cache()
            return output

        keys = cache.key_cache[module.layer_idx]
        values = cache.value_cache[module.layer_idx]

        self.compress(module, hidden_states, keys, values, output[1], kwargs)


        return output

    def post_prefilling(self, module, hidden_states, kwargs):
        cache = kwargs["past_key_value"]
        gini_tensor = torch.tensor(self.layer_gini_scores)
        normalized_scores = torch.softmax(-gini_tensor, dim=0)
        for i in range(module.config.num_hidden_layers):
            keys = cache.key_cache[i]
            values = cache.value_cache[i]

            scores = -keys.norm(dim=-1)
            
            layer_compression_ratio = self.compression_ratio * normalized_scores[i].item() * module.config.num_hidden_layers
            
            q_len = hidden_states.shape[1]
            n_kept = int(q_len * (1 - layer_compression_ratio))
            indices = scores.topk(n_kept, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

            # Prune keys and values
            keys = keys.gather(2, indices).contiguous()
            values = values.gather(2, indices).contiguous()


            cache.key_cache[i] = keys
            cache.value_cache[i] = values
