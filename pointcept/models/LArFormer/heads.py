"""
Optional supervision heads for LArFormer levels.

PerTokenClsHead: a small linear classifier attached to one level's tokens,
predicting a per-token class label. Used for:
    - deghosting (binary "real / ghost" head on the spacepoint level)
    - per-voxel slice classification (nu / cosmic / no-slice)
    - any other per-token semantic label the dataset can lift

The classifier's target is built by the loss via `build_per_level_cls_target`,
which scatter-reduces a per-spacepoint truth array onto the level's tokens
through `sp_to_level_id`. See `losses.py`.
"""

import torch
import torch.nn as nn


class PerTokenClsHead(nn.Module):
    """Linear classifier on per-token features.

    Args:
        dim:         token feature dim (matches decoder/tokenizer token_dim)
        num_classes: output class count
        hidden_dim:  optional MLP width; if None, a plain linear is used
        dropout:     applied between hidden layers when hidden_dim is set
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (M, D) → (M, num_classes)."""
        return self.net(tokens)
