"""LArFormerKeypoint — Phase-1 dense per-spacepoint keypoint-score model.

The simplest member of the LArFormer keypoint family (see
`lartpc/larformer_analysis/archive/keypoint_v1/README.md`): a (frozen) backbone +
`KeypointScoreHead` predicting the per-SP `(N, n_types)` keypoint
proximity-score field, trained against the `kpscores` GT with weighted MSE.

It deliberately has NO decoder / queries / matching — it's the dense
"heatmap" head (Design Option 1), used to (a) validate the keypoint data
path end-to-end, (b) hit the milestone of driving the score loss to ~0 on
the dev cache, and (c) later serve as a backbone-warming auxiliary signal +
anchor seeder for the query-based head (Phase 2/3).

Input data_dict (from LArFormerStage12CacheDataset / LArFormerDataset with
emit_keypoints=True), flat-batched by `larformer_collate`:
    coord_norm, feat, grid_coord, offset   — backbone inputs
    kpscores  (sum N_b, n_types)            — per-SP regression target

Forward:
    train → {"loss": total, "loss_kp": ..., "loss_kp_mse_all": ...,
             "loss_kp_mse_pos": ..., "loss_kp_frac_pos": ...}
    eval  → {"predictions": list[B] of {"coord_norm", "kpscores_pred"},
             "loss": ...(if kpscores present)}
"""

from typing import Optional

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model

from .keypoint_heads import KeypointScoreHead, KeypointOffsetHead


@MODELS.register_module()
class LArFormerKeypoint(nn.Module):
    """Dense per-SP keypoint-score model.

    Args:
        backbone:               registry config for the backbone (Sonata/PTv3).
        backbone_out_channels:  per-SP feature dim D_bb the backbone emits.
        n_keypoint_types:       output channels (default 6).
        freeze_backbone:        run backbone under no_grad + requires_grad=False.
        backbone_weight:        optional Sonata pretrain checkpoint path.
        head_hidden_dim:        KeypointScoreHead MLP width (0 = linear).
        head_dropout:           KeypointScoreHead dropout.
        loss_kind:              "mse" (default; on sigmoid(logits)) or "bce"
                                 (BCE-with-logits; note: won't reach 0 on soft
                                 targets — its minimum is the target entropy).
        pos_weight:             per-element up-weight for points with
                                 target > pos_threshold (combats the heavy
                                 zero imbalance). 1.0 = unweighted (clean for
                                 the overfit milestone).
        pos_threshold:          score above which a point counts as "positive"
                                 for pos_weight + the mse_pos diagnostic.
        weight_keypoint:        scalar on the keypoint loss in "total".
    """

    def __init__(
        self,
        backbone: dict,
        backbone_out_channels: int,
        n_keypoint_types: int = 6,
        freeze_backbone: bool = True,
        backbone_weight: Optional[str] = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        loss_kind: str = "mse",
        pos_weight: float = 1.0,
        pos_threshold: float = 0.05,
        weight_keypoint: float = 1.0,
        enable_offset_head: bool = False,
        weight_offset: float = 1.0,
        offset_supervision_threshold: float = 0.05,
        coord_scale: float = 179.55,
    ):
        super().__init__()
        if loss_kind not in ("mse", "bce"):
            raise ValueError(f"loss_kind must be 'mse' or 'bce'; got {loss_kind!r}")
        self.backbone = build_model(backbone)
        if backbone_weight is not None:
            self._load_backbone_weight(backbone_weight)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_out_channels = int(backbone_out_channels)
        self.n_keypoint_types = int(n_keypoint_types)
        self.head = KeypointScoreHead(
            in_dim=backbone_out_channels,
            n_types=n_keypoint_types,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        self.loss_kind = loss_kind
        self.pos_weight = float(pos_weight)
        self.pos_threshold = float(pos_threshold)
        self.weight_keypoint = float(weight_keypoint)
        self.enable_offset_head = bool(enable_offset_head)
        self.weight_offset = float(weight_offset)
        self.offset_supervision_threshold = float(offset_supervision_threshold)
        self.coord_scale = float(coord_scale)
        self.offset_head = (
            KeypointOffsetHead(
                in_dim=backbone_out_channels,
                n_types=n_keypoint_types,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
            if self.enable_offset_head else None
        )

    # ------------------------------------------------------------------

    def _load_backbone_weight(self, weight_path: str) -> None:
        """Load a Sonata pretrain into self.backbone (prefix-stripping +
        shape-aware filter; same convention as LArFormer._load_backbone_weight)."""
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt:
            raise KeyError(
                f"{weight_path}: no 'state_dict' key; got "
                f"{list(ckpt.keys())[:10]}"
            )
        sd = ckpt["state_dict"]
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        if sd and all(k.startswith("backbone.") for k in sd.keys()):
            sd = {k[len("backbone."):]: v for k, v in sd.items()}
        target_shapes = {k: tuple(v.shape)
                         for k, v in self.backbone.state_dict().items()}
        filtered = {}
        n_mismatch = 0
        for k, v in sd.items():
            if k not in target_shapes or tuple(v.shape) == target_shapes[k]:
                filtered[k] = v
            else:
                n_mismatch += 1
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(f"[LArFormerKeypoint] Loaded backbone weight: {weight_path}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)} "
              f"shape_mismatch_dropped={n_mismatch}")

    def _encode(self, data_dict: dict) -> torch.Tensor:
        bb_in = {
            "coord":      data_dict["coord_norm"],
            "feat":       data_dict["feat"],
            "offset":     data_dict["offset"],
            "grid_coord": data_dict["grid_coord"],
        }
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(bb_in, return_point=True)
        else:
            result = self.backbone(bb_in, return_point=True)
        return result["point"].feat

    # ------------------------------------------------------------------

    def _keypoint_loss(self, logits: torch.Tensor, target: torch.Tensor):
        """Return (total_loss, diag_dict). target is (N, n_types) in [0,1]."""
        if self.loss_kind == "bce":
            pred = torch.sigmoid(logits.detach())
            per_elem = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
        else:  # mse on sigmoid(logits)
            pred = torch.sigmoid(logits)
            per_elem = (pred - target) ** 2

        if self.pos_weight != 1.0:
            w = torch.where(
                target > self.pos_threshold,
                torch.full_like(target, self.pos_weight),
                torch.ones_like(target),
            )
            loss = (per_elem * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss = per_elem.mean()

        with torch.no_grad():
            pred_s = pred if self.loss_kind == "bce" else pred.detach()
            mse_all = ((pred_s - target) ** 2).mean()
            pos = target > self.pos_threshold
            mse_pos = (((pred_s - target) ** 2)[pos].mean()
                       if pos.any() else target.new_zeros(()))
            frac_pos = pos.float().mean()
        diag = {
            "kp_mse_all": mse_all,
            "kp_mse_pos": mse_pos,
            "kp_frac_pos": frac_pos,
        }
        return loss, diag

    def _offset_loss(self, offset_pred, offset_target, score_target):
        """Smooth-L1 offset (vote) loss, supervised only at points NEAR a
        keypoint (score_target > offset_supervision_threshold).

        offset_pred / offset_target: (N, n_types, 3) in normalized frame.
        score_target: (N, n_types). Returns (loss, diag).
        """
        mask = score_target > self.offset_supervision_threshold   # (N, n_types)
        if not bool(mask.any()):
            z = offset_pred.new_zeros(())
            return z, {"off_err_cm": z,
                       "off_n_sup": offset_pred.new_zeros(())}
        pred_m = offset_pred[mask]                                 # (M, 3)
        tgt_m = offset_target[mask].to(pred_m.dtype)
        loss = nn.functional.smooth_l1_loss(pred_m, tgt_m, reduction="mean")
        with torch.no_grad():
            # Mean residual distance in cm (denormalize by coord_scale).
            err_cm = (pred_m - tgt_m).norm(dim=-1).mean() * self.coord_scale
            n_sup = torch.tensor(float(mask.sum()), device=pred_m.device)
        return loss, {"off_err_cm": err_cm, "off_n_sup": n_sup}

    @staticmethod
    def _per_event_slices(offset: torch.Tensor):
        offs = offset.detach().cpu().tolist()
        out, prev = [], 0
        for o in offs:
            out.append(slice(prev, o))
            prev = o
        return out

    def forward(self, data_dict: dict) -> dict:
        feat = self._encode(data_dict)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1000.0, neginf=-1000.0
                                ).clamp(-1000.0, 1000.0)
        logits = self.head(feat)                       # (N_total, n_types)
        offset_pred = (self.offset_head(feat)          # (N_total, n_types, 3)
                       if self.offset_head is not None else None)

        has_target = "kpscores" in data_dict and data_dict["kpscores"] is not None
        has_off_target = (self.offset_head is not None
                          and "kpoffsets" in data_dict
                          and data_dict["kpoffsets"] is not None)

        def _loss_dict():
            """Compute the (total, components) loss dict given a target."""
            target = data_dict["kpscores"].to(logits.dtype)
            kp_loss, diag = self._keypoint_loss(logits, target)
            total = self.weight_keypoint * kp_loss
            d = {"loss_kp": kp_loss}
            for k, v in diag.items():
                d[f"loss_{k}"] = v
            if has_off_target:
                off_loss, off_diag = self._offset_loss(
                    offset_pred, data_dict["kpoffsets"], target)
                total = total + self.weight_offset * off_loss
                d["loss_off"] = off_loss
                for k, v in off_diag.items():
                    d[f"loss_{k}"] = v
            d["loss"] = total
            return d

        if self.training:
            if not has_target:
                raise KeyError(
                    "LArFormerKeypoint.forward(train): data_dict has no "
                    "'kpscores' target. Use a dataset with emit_keypoints=True."
                )
            return _loss_dict()

        # Eval / inference
        preds = []
        for sl in self._per_event_slices(data_dict["offset"]):
            entry = {
                "coord_norm": data_dict["coord_norm"][sl].detach(),
                "kpscores_pred": torch.sigmoid(logits[sl]).detach(),
            }
            if offset_pred is not None:
                entry["kpoffsets_pred"] = offset_pred[sl].detach()
                # Voted keypoint location per (SP, type): coord + offset.
                entry["kpvote_pred"] = (
                    data_dict["coord_norm"][sl].unsqueeze(1).detach()
                    + offset_pred[sl].detach())
            preds.append(entry)
        result = {"predictions": preds}
        if has_target:
            with torch.no_grad():
                result.update(_loss_dict())
        return result
