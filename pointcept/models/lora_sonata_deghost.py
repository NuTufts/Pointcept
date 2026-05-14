"""
LoRA Fine-tuning for SONATA Pretrained Backbone — De-ghosting Task
==================================================================

Implements a binary de-ghosting head with LoRA adapters, trained
independently from the SSNet particle classification head.

Two-stage inference pipeline
-----------------------------
  1. Load backbone + LoRA_deghost weights.
     Forward pass -> deghost_head -> argmax -> ghost mask.
  2. Load backbone + LoRA_ssnet weights.
     Forward pass on surviving points -> seg_head -> 8-class labels.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch


# ---------------------------------------------------------------------------
# LoRA primitives  (identical to lora_sonata.py -- kept here for
# self-containment so the two files can be used independently)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    The frozen base weight is wrapped; only lora_A and lora_B are trained.

    LoRA math:
        W' = W + scale * B @ A
        A in R^{rank x in},  B in R^{out x rank}
        init: A ~ kaiming_uniform, B = 0  ->  delta_W = 0 at step 0
        scale = alpha / rank
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

        # Frozen base weight/bias as nn.Parameters (requires_grad=False)
        # so named_parameters() can see and freeze them correctly.
        self.weight = nn.Parameter(base_linear.weight.data, requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result   = F.linear(x, self.weight, self.bias)          # frozen path
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA delta folded in (for export)."""
        merged = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
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
    Replace every nn.Linear whose local name matches any string in
    *target_modules* with a LoRALinear.  Returns the number injected.
    """
    replacements = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        local_name = full_name.split(".")[-1]
        if any(t in local_name for t in target_modules):
            parts  = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], module))

    for parent, attr, linear in replacements:
        lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        if verbose:
            print(f"  [LoRA] Injected into: {attr}  "
                  f"({linear.in_features}->{linear.out_features})  rank={rank}")
    return len(replacements)


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze all parameters; then unfreeze only the LoRA A/B matrices."""
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)


# ---------------------------------------------------------------------------
# De-ghosting segmentor
# ---------------------------------------------------------------------------

@MODELS.register_module("SonataLoRADeghostSegmentor")
class SonataLoRADeghostSegmentor(nn.Module):
    """
    SONATA backbone with LoRA adapters + binary de-ghosting head.

    Outputs (N, 2) logits -- one logit per class (0=real, 1=ghost) --
    mirroring the SSNet head structure exactly.  This is fully compatible
    with SemSegEvaluator (which calls output.max(1)[1]) and
    LovaszLoss(mode="multiclass").
    """

    def __init__(
        self,
        backbone_out_channels: int = 1232,
        backbone: dict = None,
        criteria: list = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        ghost_class_index: int = 1,
        freeze_backbone_non_lora: bool = True,
        strip_student: bool = True,
    ):
        """
        strip_student:
            When True (default), delete ``self.backbone.student`` after the
            backbone is built. The Sonata-v1m1 wrapper holds two copies of
            the encoder for DINO-style self-distillation: ``self.teacher``
            (the EMA copy, used in fine-tuning forward path via
            ``return_point=True``) and ``self.student``. For downstream
            fine-tuning the student is never called in forward and never
            receives gradients — it's purely dead weight in GPU memory
            (~half of the backbone parameters). Stripping it here, BEFORE
            ``build_optimizer`` walks ``named_parameters()``, also keeps
            those frozen parameters out of the optimizer's param_groups
            (otherwise the optimizer would hold tensor references and the
            memory wouldn't actually be freed). The accompanying student
            keys in the pretraining checkpoint will then be reported as
            "unexpected" by the checkpoint loader — harmless.

            Set to False if you want to keep the student subtree around
            for debugging or ablation (e.g., comparing teacher-vs-student
            features at the same input).
        """
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["qkv", "proj"]

        self.ghost_class_index = ghost_class_index

        # ---- Build backbone -----------------------------------------------
        from pointcept.models import build_model
        from pointcept.utils.config import Config
        self.backbone = build_model(Config(backbone))

        # ---- Strip the student subtree -------------------------------------
        # Done BEFORE LoRA injection so we don't inject (and then leak) a
        # bunch of student-side LoRA adapters that never see gradient flow.
        if strip_student and hasattr(self.backbone, "student"):
            n_student_params = sum(
                p.numel() for p in self.backbone.student.parameters()
            )
            print(f"\n[DeghostSegmentor] Stripping student subtree "
                  f"({n_student_params:,} params freed). Teacher subtree "
                  f"is the one used in forward(return_point=True).")
            del self.backbone.student

        # ---- Inject LoRA ---------------------------------------------------
        print(f"\n{'='*60}")
        print(f"[DeghostSegmentor] Injecting LoRA (rank={lora_rank}, "
              f"alpha={lora_alpha}) into: {lora_target_modules}")
        n_injected = inject_lora(
            self.backbone,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            verbose=True,
        )
        print(f"  Total adapters injected: {n_injected}")

        # ---- Freeze base backbone weights ---------------------------------
        if freeze_backbone_non_lora:
            freeze_non_lora(self.backbone)
            print("  Base backbone weights: FROZEN")
            print("  LoRA A/B matrices:     TRAINABLE")

        # ---- Binary de-ghosting head (fully trainable) --------------------
        # (N, 2) output: [real_logit, ghost_logit]
        # Compatible with SemSegEvaluator.output.max(1)[1] and
        # LovaszLoss(mode="multiclass").
        self.deghost_head = nn.Linear(backbone_out_channels, 2)

        # ---- Loss ---------------------------------------------------------
        self.criteria = build_criteria(criteria)

        # ---- Summary ------------------------------------------------------
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total params   : {total_params:,}")
        print(f"  Trainable      : {train_params:,}  "
              f"({100. * train_params / total_params:.2f}%)")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------

    def forward(self, data_dict: dict) -> dict:
        backbone_out = self.backbone(data_dict, return_point=True)
        point  = backbone_out["point"]
        feat   = point.feat                          # (N, backbone_out_channels)
        logits = self.deghost_head(feat)             # (N, 2)

        if self.training:
            output = {"loss": self.criteria(logits, data_dict["segment"])}
        else:
            output = {"seg_logits": logits}
            if "segment" in data_dict:
                output["loss"] = self.criteria(logits, data_dict["segment"])
        return output

    # -----------------------------------------------------------------------

    def predict_ghost_mask(self, data_dict: dict) -> torch.BoolTensor:
        """
            ghost_mask = deghost_model.predict_ghost_mask(data_dict)
            real_data  = filter_points(data_dict, ~ghost_mask)
            ssnet_out  = ssnet_model(real_data)
        """
        self.eval()
        with torch.no_grad():
            backbone_out = self.backbone(data_dict, return_point=True)
            feat   = backbone_out["point"].feat
            logits = self.deghost_head(feat)             # (N, 2)
            labels = logits.argmax(dim=-1)               # (N,)
        return labels == self.ghost_class_index          # True where ghost

    # -----------------------------------------------------------------------

    def get_trainable_parameters(self):
       
        lora_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        ]
        head_params = list(self.deghost_head.parameters())
        return lora_params, head_params

    # -----------------------------------------------------------------------

    def merge_lora_weights(self) -> None:
        """
        Fold all LoRA adapters into their base weights in-place.
        Call before exporting a merged checkpoint for fast inference.
        """
        replaced = []
        for full_name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                parts  = full_name.split(".")
                parent = self
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                replaced.append((parent, parts[-1], module))

        for parent, attr, lora_linear in replaced:
            setattr(parent, attr, lora_linear.merge_weights())
        print(f"[DeghostSegmentor] Merged {len(replaced)} LoRA adapters.")
