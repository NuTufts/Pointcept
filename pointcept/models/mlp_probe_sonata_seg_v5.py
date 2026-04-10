"""
Parameter-Matched MLP Probe for SONATA on LArTPC
=================================================

Implements a frozen-backbone + MLP-head segmentor whose trainable
parameter count is matched to the LoRA fine-tuning baseline.

Design Comments
----------------
MLP Linear Prob run to compare as baseline finetuning method against LoRA
The LoRA run (rank=16, 92 adapters) has ~1,215,394 trainable parameters.

Only the head parameters are trainable; the backbone is fully frozen via
requires_grad_(False).  This means the comparison is:

    Linear probe     : frozen backbone  +  linear head         (~10k params)
    MLP probe (this) : frozen backbone  +  MLP head         (~1.21M params)
    LoRA fine-tuning : frozen backbone  +  LoRA adapters + linear head (~1.21M params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch


@MODELS.register_module("SonataMLPProbeSegmentor")
class SonataMLPProbeSegmentor(nn.Module):
    """
    SONATA backbone (fully frozen) + parameter-matched MLP segmentation head.

    The MLP head has a single hidden layer sized so that the total trainable
    parameter count equals the LoRA fine-tuning baseline (~1.215M).

    Parameters
    ----------
    num_classes : int
        Number of semantic classes.
    backbone_out_channels : int
        Feature dimension output by the backbone (e.g. 1232 for
        up_cast_level=4 with enc_channels=(48,96,192,384,512)).
    backbone : dict
        Config dict for the SONATA backbone (type="Sonata-v1m1").
    criteria : list[dict]
        Loss configs (passed to build_criteria).
    hidden_dim : int
        Hidden layer width.  Default 979 gives ~1.215M trainable params
        when backbone_out_channels=1232, num_classes=8, matching the
        LoRA run with rank=16 and 92 injected adapters.
    dropout : float
        Dropout probability after the hidden activation.  Default 0.1.
    activation : str
        Hidden activation: "gelu" (default) or "relu"(claude recommended gelu)
    class_priors : list[float], optional
        Per-class prior probabilities used to initialise the output layer
        bias as log(p / (1-p)).  Speeds up convergence on imbalanced data.
    freeze_backbone : bool
        If True (default), all backbone parameters are frozen.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        backbone: dict,
        criteria: list,
        hidden_dim: int = 979,
        dropout: float = 0.1,
        activation: str = "gelu",
        class_priors: Optional[List[float]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ---- Build backbone -----------------------------------------------
        from pointcept.models import build_model
        from pointcept.utils.config import Config
        self.backbone = build_model(Config(backbone))

        # ---- Freeze backbone ----------------------------------------------
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # ---- MLP head (fully trainable) -----------------------------------
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_dim),
            act_fn,
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialise output bias from class priors
        # (the first linear uses default kaiming init)
        if class_priors is not None:
            assert len(class_priors) == num_classes, (
                f"class_priors length {len(class_priors)} != num_classes {num_classes}"
            )
            out_linear = self.seg_head[-1]
            with torch.no_grad():
                priors = torch.tensor(class_priors, dtype=torch.float32)
                priors = priors.clamp(1e-6, 1.0 - 1e-6)
                out_linear.bias.copy_(torch.log(priors / (1.0 - priors)))
            print("  Head output bias: initialised from log-prior")

        # ---- Loss ---------------------------------------------------------
        self.criteria = build_criteria(criteria)

        # ---- Summary ------------------------------------------------------
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_params  = sum(p.numel() for p in self.seg_head.parameters())
        print(f"\n{'='*60}")
        print(f"SonataMLPProbeSegmentor")
        print(f"  Architecture : {backbone_out_channels} → {hidden_dim} → {num_classes}")
        print(f"  Activation   : {activation.upper()},  Dropout: {dropout}")
        print(f"  Backbone     : {'FROZEN' if freeze_backbone else 'TRAINABLE'}")
        print(f"  Head params  : {head_params:,}")
        print(f"  Total params : {total_params:,}")
        print(f"  Trainable    : {train_params:,}  "
              f"({100.0 * train_params / total_params:.2f}%)")
        print(f"  LoRA target  : 1,215,394  (Δ = {train_params - 1_215_394:+,})")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    def forward(self, data_dict: dict) -> dict:
        # Run frozen backbone
        backbone_out = self.backbone(data_dict, return_point=True)
        point = backbone_out["point"]
        feat  = point.feat                  # (N, backbone_out_channels)

        logits = self.seg_head(feat)        # (N, num_classes)

        if self.training:
            output = {"loss": self.criteria(logits, data_dict["segment"])}
        else:
            output = {"seg_logits": logits}
            if "segment" in data_dict:
                output["loss"] = self.criteria(logits, data_dict["segment"])
        return output
