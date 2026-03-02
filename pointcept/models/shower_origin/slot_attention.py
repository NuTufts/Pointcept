"""
Slot Attention Module for Shower Origin Prediction

Implements the Slot Attention mechanism from Locatello et al. (2020)
"Object-Centric Learning with Slot Attention"

Slot Attention learns to aggregate a set of input features into a fixed
number of slot representations through iterative attention with competition
between slots.

Key insight: Softmax is applied over slots (not inputs), making slots
compete for each input point. This encourages specialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention module for aggregating point features into K slot vectors.

    Given N input point features, produces K slot vectors that capture
    different aspects of the input. Slots compete for inputs through
    softmax normalization over the slot dimension.

    Args:
        num_slots: Number of output slot vectors (K)
        dim: Feature dimension (D)
        num_iterations: Number of iterative refinement steps (default: 3)
        hidden_dim: Hidden dimension for MLP refinement (default: dim * 2)
        eps: Epsilon for numerical stability (default: 1e-8)

    Input:
        inputs: (B, N, D) tensor of N point features

    Output:
        slots: (B, K, D) tensor of K slot vectors
        attn: (B, K, N) attention weights (which inputs each slot attends to)
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        num_iterations: int = 3,
        hidden_dim: int = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        if hidden_dim is None:
            hidden_dim = dim * 2

        # Learnable slot initialization parameters
        # Slots are sampled from N(mu, sigma) at the start of each forward pass
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        # Attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # GRU for slot state updates
        self.gru = nn.GRUCell(dim, dim)

        # MLP for slot refinement (applied after GRU update)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # Layer normalizations
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # Initialize slot parameters
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.constant_(self.slots_log_sigma, -2.0)  # Start with small sigma

    def forward(self, inputs, num_slots=None):
        """
        Forward pass of Slot Attention.

        Args:
            inputs: (B, N, D) input point features
            num_slots: Optional override for number of slots

        Returns:
            slots: (B, K, D) slot vectors
            attn: (B, K, N) final attention weights
        """
        B, N, D = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots

        # Normalize inputs once (they don't change during iterations)
        inputs = self.norm_inputs(inputs)

        # Compute keys and values from inputs (also constant during iterations)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)

        # Initialize slots by sampling from learned distribution
        # Different samples per batch element for diversity
        sigma = self.slots_log_sigma.exp()
        slots = self.slots_mu + sigma * torch.randn(
            B, K, D, device=inputs.device, dtype=inputs.dtype
        )

        # Iterative refinement
        for iteration in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute queries from current slot states
            q = self.to_q(slots)  # (B, K, D)

            # Attention scores: (B, K, N)
            # Each slot queries all input points
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn_logits = attn_logits.clamp(-50, 50)

            # KEY DIFFERENCE FROM STANDARD ATTENTION:
            # Softmax over slots (dim=1), not over inputs
            # This makes slots COMPETE for each input point
            attn = F.softmax(attn_logits, dim=1)  # (B, K, N)

            # Normalize by number of points attending to each slot
            # This prevents slots with many points from having huge updates
            attn_normalized = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            # Weighted aggregation of values
            updates = torch.einsum('bkn,bnd->bkd', attn_normalized, v)  # (B, K, D)

            # GRU update: combine previous slot state with new information
            # Reshape for GRU: (B*K, D)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, K, D)

            # MLP refinement with residual connection
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class MultiHeadSlotAttention(nn.Module):
    """
    Multi-head variant of Slot Attention.

    Runs multiple independent slot attention heads and concatenates results,
    similar to multi-head attention in transformers.

    Args:
        num_slots: Number of slots per head
        dim: Total feature dimension (will be split across heads)
        num_heads: Number of attention heads
        num_iterations: Iterations per head
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        num_heads: int = 4,
        num_iterations: int = 3,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Independent slot attention for each head
        self.heads = nn.ModuleList([
            SlotAttention(
                num_slots=num_slots,
                dim=self.head_dim,
                num_iterations=num_iterations,
            )
            for _ in range(num_heads)
        ])

        # Project inputs to per-head dimension
        self.input_proj = nn.Linear(dim, dim)

        # Project concatenated outputs back to original dimension
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, inputs):
        """
        Args:
            inputs: (B, N, D) input features

        Returns:
            slots: (B, K, D) slot vectors
            attn: (B, num_heads, K, N) attention weights per head
        """
        B, N, D = inputs.shape

        # Project and reshape for heads
        inputs = self.input_proj(inputs)
        inputs = inputs.reshape(B, N, self.num_heads, self.head_dim)
        inputs = inputs.permute(2, 0, 1, 3)  # (num_heads, B, N, head_dim)

        # Run each head independently
        all_slots = []
        all_attn = []
        for h, head in enumerate(self.heads):
            slots_h, attn_h = head(inputs[h])
            all_slots.append(slots_h)
            all_attn.append(attn_h)

        # Concatenate along feature dimension
        slots = torch.cat(all_slots, dim=-1)  # (B, K, D)
        slots = self.output_proj(slots)

        # Stack attention weights
        attn = torch.stack(all_attn, dim=1)  # (B, num_heads, K, N)

        return slots, attn
