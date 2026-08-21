from __future__ import annotations

import torch
from torch import nn


class DeepSetValueModel(nn.Module):
    """Permutation-invariant candidate-set value surrogate with context conditioning."""

    def __init__(
        self,
        member_features: int,
        context_features: int,
        hidden_features: int = 32,
    ) -> None:
        super().__init__()
        self.member_encoder = nn.Sequential(
            nn.Linear(member_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_features + context_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(
        self,
        members: torch.Tensor,
        member_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if members.ndim != 3 or member_mask.ndim != 2:
            raise ValueError("members must be [batch, set, feature] and mask [batch, set]")
        if members.shape[:2] != member_mask.shape:
            raise ValueError("member mask does not match member tensor")
        encoded = self.member_encoder(members)
        mask = member_mask.unsqueeze(-1)
        counts = mask.sum(dim=1).clamp_min(1)
        pooled_mean = (encoded * mask).sum(dim=1) / counts
        pooled_max = encoded.masked_fill(~mask, float("-inf")).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        combined = torch.cat([pooled_mean, pooled_max, context], dim=1)
        return self.value_head(combined).squeeze(1)
