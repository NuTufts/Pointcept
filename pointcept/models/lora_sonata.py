"""
LoRA Fine-tuning for SONATA Pretrained Backbone
================================================

Implements Low-Rank Adaptation (LoRA) for the PT-v3m2 encoder inside
the SONATA self-supervised backbone.

Architecture
------------
The SonataLoRASegmentor wraps the full SONATA model (same as the linear
probe SonataSegmentor), but instead of freezing all backbone parameters
it:
  1. Freezes every parameter in the backbone.
  2. Injects trainable LoRA A/B matrices into every nn.Linear in the
     specified target modules (by default the QKV + projection linears
     inside each SerializedAttention block).
  3. Keeps the segmentation head fully trainable.

During training, only LoRA A/B matrices and the head parameters receive
gradients, giving a very small trainable parameter count while allowing
the attention layers to adapt to the LArTPC domain.
"""

import math
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch


# ---------------------------------------------------------------------------
# LoRA primitives
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    The frozen base weight is wrapped; only lora_A and lora_B are trained.
    The merged weight W + B@A is used both for forward and for export
    (call merge_weights() to fold into a plain nn.Linear).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank         = rank
        self.scale        = alpha / rank

        # Store base weight/bias as frozen nn.Parameters so that:
        #   (a) named_parameters() can see and freeze them,
        #   (b) state_dict() / load_state_dict() work correctly for checkpointing,
        #   (c) they move with .to(device) / .half() etc.
        # requires_grad=False keeps them out of the gradient graph.
        self.weight = nn.Parameter(base_linear.weight.data, requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA matrices  (always float32 for stability)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # kaiming-uniform init for A, zero init for B  → ΔW=0 at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen weight, no grad)
        result = F.linear(x, self.weight, self.bias)
        # LoRA delta
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA delta folded in (for inference export)."""
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias is not None,
                           device=self.weight.device,
                           dtype=self.weight.dtype)
        with torch.no_grad():
            delta = self.scale * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.weight.data + delta)
            if self.bias is not None:
                merged.bias.copy_(self.bias.data)
        return merged

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float,
    verbose: bool = True,
) -> int:
    """
    Walk the module tree and replace every nn.Linear whose name (relative
    to the root) contains any string in *target_modules* with a LoRALinear.
    """
    injected = 0
    # collect (parent_module, child_name, child_module) to avoid
    # mutating the tree while iterating
    replacements = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if this linear's local name (last segment) matches a target
        local_name = full_name.split(".")[-1]
        if any(t in local_name for t in target_modules):
            # Navigate to parent
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], module))

    for parent, attr, linear in replacements:
        lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        injected += 1
        if verbose:
            print(f"  [LoRA] Injected into: {attr}  "
                  f"({linear.in_features}→{linear.out_features})  rank={rank}")
    return injected


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze all parameters except LoRA A/B 
    """
    # Pass 1: freeze everything
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    # Pass 2: un-freeze only LoRA adapter matrices
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Segmentor
# ---------------------------------------------------------------------------

@MODELS.register_module("SonataLoRASegmentor")
class SonataLoRASegmentor(nn.Module):
    """
    SONATA backbone with LoRA adapters + segmentation head.

    Compared to SonataSegmentor (linear probe), this model:
      - Loads the same pretrained SONATA checkpoint.
      - Injects trainable LoRA adapters into the attention linears.
      - Trains LoRA adapters + segmentation head 
    Parameters
    ----------
    num_classes : int
        Number of semantic classes.
    backbone_out_channels : int
        Feature dimension output by the backbone (matches up_cast_level
        channel concatenation, e.g. 1232 for up_cast_level=4).
    backbone : dict
        Config dict for the SONATA backbone (type="Sonata-v1m1").
    criteria : list[dict]
        Loss configs (passed to build_criteria).
    lora_rank : int
        LoRA rank r.  Typical values: 4, 8, 16, 32.
    lora_alpha : float
        LoRA scaling α.  Effective scale = α/r.
    lora_dropout : float
        Dropout applied to the LoRA path.
    lora_target_modules : list[str]
        Sub-strings matched against the *local* name of each nn.Linear
        in the backbone.  Defaults to ["qkv", "proj"] which targets
        the QKV projection and output projection in each attention block.
    class_priors : list[float], optional
        Per-class prior probabilities used to initialise the head bias as
        log(p / (1-p)).  Speeds up early convergence on imbalanced data.
    freeze_backbone_non_lora : bool
        If True (default), all backbone parameters outside the LoRA
        adapters are frozen.  Set False for full fine-tuning baseline.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        backbone: dict,
        criteria: list,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        class_priors: Optional[List[float]] = None,
        freeze_backbone_non_lora: bool = True,
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["qkv", "proj"]

        # ---- Build backbone -----------------------------------------------
        from pointcept.models import build_model
        from pointcept.utils.config import Config
        self.backbone = build_model(Config(backbone))

        # ---- Inject LoRA into backbone attention linears ------------------
        print(f"\n{'='*60}")
        print(f"Injecting LoRA adapters (rank={lora_rank}, α={lora_alpha}) "
              f"into modules matching: {lora_target_modules}")
        n_injected = inject_lora(
            self.backbone,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            verbose=True,
        )
        print(f"  Total adapters injected: {n_injected}")

        # ---- Freeze base weights, keep LoRA trainable ---------------------
        if freeze_backbone_non_lora:
            freeze_non_lora(self.backbone)
            print("  Base backbone weights: FROZEN")
            print("  LoRA A/B matrices:     TRAINABLE")

        # ---- Segmentation head (fully trainable) --------------------------
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

        if class_priors is not None:
            assert len(class_priors) == num_classes, (
                f"class_priors length {len(class_priors)} != num_classes {num_classes}"
            )
            with torch.no_grad():
                priors = torch.tensor(class_priors, dtype=torch.float32)
                priors = priors.clamp(1e-6, 1 - 1e-6)
                self.seg_head.bias.copy_(torch.log(priors / (1.0 - priors)))
            print("  Head bias: initialised from log-prior")

        # ---- Loss ---------------------------------------------------------
        self.criteria = build_criteria(criteria)

        # ---- Summary ------------------------------------------------------
        total_params  = sum(p.numel() for p in self.parameters())
        train_params  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total params   : {total_params:,}")
        print(f"  Trainable      : {train_params:,}  "
              f"({100. * train_params / total_params:.2f}%)")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    #def forward(self, data_dict: dict) -> dict:
        # Run backbone (up_cast gives per-point features at original resolution)
    #    point = self.backbone(data_dict)

        # point.feat shape: (N_total_voxels, backbone_out_channels)
    #    feat   = point.feat
    #    logits = self.seg_head(feat)  # (N, num_classes)

    #    output = {"seg_logits": logits}
    #    if "segment" in data_dict:
    #        output["loss"] = self.criteria(logits, data_dict["segment"])
    #    return output
  

    def forward(self, data_dict: dict) -> dict:
        backbone_out = self.backbone(data_dict, return_point=True)
        point = backbone_out["point"]
        feat   = point.feat
        logits = self.seg_head(feat)

        if self.training:
        # During training: only return loss (InformationWriter logs all keys as scalars)
            output = {"loss": self.criteria(logits, data_dict["segment"])}
        else:
        # During evaluation: return seg_logits for the evaluator
            output = {"seg_logits": logits}
            if "segment" in data_dict:
                output["loss"] = self.criteria(logits, data_dict["segment"])
        return output

  
    # -----------------------------------------------------------------------
    def get_trainable_parameters(self):
        """
        Return two groups for the optimizer:
          - lora: LoRA A/B matrices (higher lr, no weight decay typical)
          - head: segmentation head weights/bias
        """
        lora_params = [p for n, p in self.named_parameters()
                       if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        head_params = list(self.seg_head.parameters())
        return lora_params, head_params

    # -----------------------------------------------------------------------
    def merge_lora_weights(self) -> None:
        """
        Fold all LoRA adapters into their base weights (in-place).
        Call this before exporting a merged checkpoint for fast inference.
        After merging, the model behaves like a plain fine-tuned model.
        """
        replaced = []
        for full_name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                parts = full_name.split(".")
                parent = self
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                replaced.append((parent, parts[-1], module))

        for parent, attr, lora_linear in replaced:
            merged = lora_linear.merge_weights()
            setattr(parent, attr, merged)
        print(f"Merged {len(replaced)} LoRA adapters into base weights.")
