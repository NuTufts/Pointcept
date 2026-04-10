
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch


@MODELS.register_module("SonataMLPProbeDeghostSegmentor")
class SonataMLPProbeDeghostSegmentor(nn.Module):
    """
    SONATA backbone (fully frozen) + parameter-matched MLP de-ghosting head.

    Outputs (N, 2) logits — [real_logit, ghost_logit] — identical in shape
    to SonataLoRADeghostSegmentor, ensuring drop-in compatibility with all
    downstream evaluators, loss functions, and logging hooks.

    Parameters
    ----------
    backbone_out_channels : int
        Feature dimension output by the backbone.
        Default 1232 matches up_cast_level=4 with
        enc_channels=(48, 96, 192, 384, 512).
    backbone : dict
        Config dict for the SONATA backbone (type="Sonata-v1m1").
    criteria : list[dict]
        Loss configs. Recommended:
            LovaszLoss(mode="multiclass", ignore_index=-1)
        passed to build_criteria().
    hidden_dim : int
        Width of the single hidden layer.  Default 986 gives ~1.2178M
        trainable parameters matching the LoRA de-ghosting baseline.
    dropout : float
        Dropout probability applied after the hidden activation.
        Default 0.1.  Set to 0.0 to disable.
    activation : str
        Hidden-layer activation: "gelu" (default) or "relu".
    ghost_class_index : int
        Label value for the ghost class (default 1, after RemapGhostLabel).
        Used by predict_ghost_mask() and two-stage inference pipelines.
    freeze_backbone : bool
        If True (default), requires_grad_(False) is called on the entire
        backbone module.  The seg_head is never touched by this call, so
        its parameters always remain trainable.
    """

    def __init__(
        self,
        backbone_out_channels: int = 1232,
        backbone: dict = None,
        criteria: list = None,
        hidden_dim: int = 986,
        dropout: float = 0.1,
        activation: str = "gelu",
        ghost_class_index: int = 1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.ghost_class_index = ghost_class_index

        # ------------------------------------------------------------------ #
        # 1.  Build backbone                                                  #
        # ------------------------------------------------------------------ #
        from pointcept.models import build_model
        from pointcept.utils.config import Config
        self.backbone = build_model(Config(backbone))

        # ------------------------------------------------------------------ #
        # 2.  Freeze backbone                                                 #
        # --------------------------------------------------------------------#

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # 3.  MLP head (fully trainable)                                      #
        #                                                                     #
        # Architecture: backbone_out_channels → hidden_dim → 2               #
        # With default hidden_dim=986:                                        #
        #   1232×986 + 986 + 986×2 + 2 = 1,215,352 + 986 + 1972 + 2         #
        #                               = 1,217,812  params                   #
        # ------------------------------------------------------------------ #
        act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_dim),
            act_fn,
            dropout_layer,
            nn.Linear(hidden_dim, 2),          # 2 classes: real / ghost
        )

        # ------------------------------------------------------------------ #
        # 4.  Loss                                                            #
        # ------------------------------------------------------------------ #
        self.criteria = build_criteria(criteria)

       
        self._verify_freeze()
        self._print_summary(backbone_out_channels, hidden_dim, activation,
                            dropout, freeze_backbone)

    # ---------------------------------------------------------------------- #
    # Freeze verification                                                     #
    # ---------------------------------------------------------------------- #

    def _verify_freeze(self) -> None:
        """
        Assert that:
          (a) no backbone parameter is trainable, and
          (b) every seg_head parameter is trainable.

        Raises RuntimeError with a diagnostic message if either condition
        is violated, so misconfiguration is caught at construction time
        rather than silently producing wrong gradients.
        """
        frozen_in_head   = [n for n, p in self.seg_head.named_parameters()
                             if not p.requires_grad]
        trainable_in_bb  = [n for n, p in self.backbone.named_parameters()
                             if p.requires_grad]

        errors = []
        if frozen_in_head:
            errors.append(
                f"  HEAD parameters that are NOT trainable (should be 0):\n"
                + "\n".join(f"    {n}" for n in frozen_in_head)
            )
        if trainable_in_bb:
            errors.append(
                f"  BACKBONE parameters that ARE trainable (should be 0):\n"
                + "\n".join(f"    {n}" for n in trainable_in_bb[:10])
                + (f"\n    ... and {len(trainable_in_bb)-10} more"
                   if len(trainable_in_bb) > 10 else "")
            )
        if errors:
            raise RuntimeError(
                "SonataMLPProbeDeghostSegmentor: freeze verification FAILED!\n"
                + "\n".join(errors)
            )

    # ---------------------------------------------------------------------- #
    # Summary                                                                 #
    # ---------------------------------------------------------------------- #

    def _print_summary(self, in_dim, hidden, activation, dropout, frozen):
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        head_params  = sum(p.numel() for p in self.seg_head.parameters())
        lora_target  = 1_217_860   # LoRA A/B (1,215,394) + deghost_head (2,466)

        print(f"\n{'='*62}")
        print(f"SonataMLPProbeDeghostSegmentor")
        print(f"  Architecture : {in_dim} → {hidden} → 2  (real / ghost)")
        print(f"  Activation   : {activation.upper()},  Dropout: {dropout}")
        print(f"  Backbone     : {'FROZEN' if frozen else 'TRAINABLE'}")
        print(f"  Head params  : {head_params:,}")
        print(f"  Total params : {total_params:,}")
        print(f"  Trainable    : {train_params:,}  "
              f"({100.0 * train_params / total_params:.3f}%)")
        print(f"  LoRA target  : {lora_target:,}  "
              f"(Δ = {train_params - lora_target:+,})")
        print(f"{'='*62}\n")

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(self, data_dict: dict) -> dict:
        """
        Parameters
        ----------
        data_dict : dict
            Must contain the point-cloud fields expected by the SONATA
            backbone.  During training, must also contain "segment" with
            integer ghost labels (0=real, 1=ghost, -1=ignore).
        """
        # Run frozen backbone (no grad through backbone weights)
        with torch.no_grad():
            backbone_out = self.backbone(data_dict, return_point=True)

        feat   = backbone_out["point"].feat          # (N, backbone_out_channels)
        logits = self.seg_head(feat)                 # (N, 2)

        output: dict = {}
        if not self.training:
            output["seg_logits"] = logits

        if self.training or "segment" in data_dict:
            output["loss"] = self.criteria(logits, data_dict["segment"])

        return output

    # ---------------------------------------------------------------------- #
    # Convenience inference method (mirrors LoRA baseline API)               #
    # ---------------------------------------------------------------------- #

    def predict_ghost_mask(self, data_dict: dict) -> torch.BoolTensor:
        """
        Return a boolean mask (True = ghost) over all input points.

        Mirrors SonataLoRADeghostSegmentor.predict_ghost_mask() so the two
        models are interchangeable in a two-stage inference pipeline:

            ghost_mask = model.predict_ghost_mask(data_dict)
            real_data  = filter_points(data_dict, ~ghost_mask)
            ssnet_out  = ssnet_model(real_data)
        """
        self.eval()
        with torch.no_grad():
            backbone_out = self.backbone(data_dict, return_point=True)
            feat   = backbone_out["point"].feat
            logits = self.seg_head(feat)             # (N, 2)
            labels = logits.argmax(dim=-1)           # (N,)
        return labels == self.ghost_class_index      # BoolTensor
