from dataclasses import dataclass

import torch
from torch import nn
import numpy as np

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class AdapPress(ScorerPress):

    compression_ratio: float = 0.0
    layer_gini_scores: list = None
    num_layers: int = None

    def __post_init__(self):
        super().__post_init__()
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
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        
        # Initialize layer_gini_scores if first layer
        if module.layer_idx == 0:
            self.num_layers = module.config.num_hidden_layers
            self.layer_gini_scores = [0.0] * self.num_layers

        # Calculate and update gini score for current layer
        gini = self.calculate_gini(attentions)
        self.layer_gini_scores[module.layer_idx] = gini
        #
        # # Calculate layer-specific compression ratio based on softmax of gini scores
        # if all(score != 0.0 for score in self.layer_gini_scores):
        #     # Convert gini scores to tensor and apply softmax
        #     gini_tensor = torch.tensor(self.layer_gini_scores, device=keys.device)
        #     normalized_scores = torch.softmax(gini_tensor, dim=0)
        #     
        #     # Allocate compression ratio based on normalized scores
        #     layer_compression_ratio = self.compression_ratio * normalized_scores[module.layer_idx].item()
        #     
        #     # Calculate number of tokens to keep
        #     q_len = hidden_states.shape[1]
        #     n_kept = int(q_len * (1 - layer_compression_ratio))
        #     
        #     # Generate scores for this layer
        #     scores = torch.ones_like(keys[..., 0])
        #     if n_kept < q_len:
        #         # Set scores for tokens to be pruned to 0
        #         scores[:, :, n_kept:] = 0
        # else:
        #     # If we haven't seen all layers yet, use uniform compression
        scores = torch.ones_like(keys[..., 0])
        
        return scores

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

        


