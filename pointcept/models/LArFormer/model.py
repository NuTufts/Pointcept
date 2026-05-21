"""
LArFormer — top-level Mask2Former-style model with configurable level list.

Wires together (in this order):
    1. Backbone (built via the MODELS registry; usually Sonata/PTv3)
    2. CompositeTokenizer (one LevelBuilder per declared level)
    3. Mask2FormerDecoder (named-scale cross-attention rotation)
    4. PerTokenClsHead per level that declares supervision.cls
    5. LArFormerLoss (set loss + per-level mask aux + per-level cls)

Forward semantics — Pointcept-style:

    out = model(batched_data_dict)
    if model.training:
        loss = out["loss"]
        # plus per-component scalars: loss_cls, loss_mask_primary, loss_dice,
        # loss_aux_mask_<lvl>, loss_cls_<lvl>, loss_origin, ...
    else:
        out["predictions"]  # list[B] of dicts (see decoder + per-level cls)

The backbone runs once on the flat-batched (sum N_b, ...) input dict; the
tokenizer / decoder / loss are applied per event because Hungarian matching
is per-event. Aggregation is mean across events.

Input data_dict shape — for P1 this mirrors `ShowerClusteringDataset`'s
`shower_clustering_collate` output, so the model can be sanity-tested on
existing data before `LArFormerDataset` (P4) lands:

    coord, coord_norm, grid_coord, feat            flat (sum N_b, ...)
    offset            (B,)                          cumulative SP counts
    gt_instances_per_event                          list[B] of list[K] of dict
    per-SP truth fields (hasmatch, origin_label, ...) — flat per-SP, if
        any level declares supervision.cls with that label_src
    voxel_id / voxel_keys / voxel_offset / fragment_indices_per_event —
        accepted but ignored by P1 builders (spacepoint, voxel)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model

from .builders import LevelOutput
from .decoder import Mask2FormerDecoder
from .heads import PerTokenClsHead
from .losses import LArFormerLoss
from .refiners import build_token_refiner
from .tokenizer import CompositeTokenizer


@MODELS.register_module()
class LArFormer(nn.Module):
    """Top-level LArFormer model.

    Args:
        backbone: dict — registry config for the backbone (e.g. Sonata).
        backbone_out_channels: backbone feature dim D_bb.
        levels: list of level config dicts. Each entry must have:
            name        — unique stable key
            builder     — BUILDERS registry key
            builder_cfg — kwargs forwarded to the builder ctor (default {})
            supervision — optional; sub-keys: mask, cls (see docs/LArFormer.md)
        scale_pattern: list of level names — one per decoder layer
        token_dim: decoder token dim (must match every builder's out)
        num_queries: Q
        num_classes: total slots incl. no_object (last)
        freeze_backbone: wrap backbone forward in no_grad
        enable_origin_head: keep the per-query origin regression head
        loss_kwargs: forwarded to LArFormerLoss (num_classes is auto-set)
        decoder_kwargs: forwarded to Mask2FormerDecoder
    """

    def __init__(
        self,
        backbone: dict,
        backbone_out_channels: int,
        levels: Sequence[dict],
        scale_pattern: Sequence[str] = (),
        token_dim: int = 256,
        num_queries: int = 64,
        num_classes: int = 6,
        freeze_backbone: bool = True,
        enable_origin_head: bool = True,
        loss_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        token_refiner: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_out_channels = int(backbone_out_channels)
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.enable_origin_head = bool(enable_origin_head)

        self.tokenizer = CompositeTokenizer(
            levels_cfg=levels,
            in_dim=backbone_out_channels,
            token_dim=token_dim,
        )
        self.levels_cfg = list(self.tokenizer.levels_cfg)

        # TokenRefiner sits between the tokenizer (static pooled features)
        # and the decoder. Default = IdentityRefiner (zero-op, reproduces
        # pre-refiner behavior). See pointcept/models/LArFormer/refiners/
        # for available implementations.
        refiner_cfg = dict(token_refiner) if token_refiner is not None else None
        if refiner_cfg is not None \
                and refiner_cfg.get("type") not in (None, "IdentityRefiner"):
            # Convenience: refiners that need to know token dim + the level
            # list get them auto-injected here, so the config doesn't have
            # to repeat token_dim / levels. `levels_cfg` is what lets the
            # refiner build all per-level submodules EAGERLY at __init__
            # — required for DDP and for model.to(device) to work.
            refiner_cfg.setdefault("dim", token_dim)
            refiner_cfg.setdefault("levels_cfg", self.levels_cfg)
        self.token_refiner = build_token_refiner(refiner_cfg)

        # When num_queries == 0 the model degenerates to a pure per-level
        # cls model (Stage-1 deghoster pattern: no instance reasoning).
        # We skip the decoder entirely; the per-level cls heads below carry
        # the whole supervision load. The Mask2Former path is bypassed in
        # forward() and in LArFormerLoss when decoder_output is None.
        if self.num_queries > 0:
            decoder_kwargs = dict(decoder_kwargs or {})
            self.decoder = Mask2FormerDecoder(
                dim=token_dim,
                scale_pattern=list(scale_pattern),
                num_queries=num_queries,
                num_classes=num_classes,
                enable_origin_head=self.enable_origin_head,
                **decoder_kwargs,
            )
        else:
            if scale_pattern:
                raise ValueError(
                    f"num_queries=0 disables the decoder; "
                    f"scale_pattern must be empty (got {list(scale_pattern)!r})"
                )
            self.decoder = None

        # Per-level cls heads (one per level with supervision.cls declared).
        self.cls_heads = nn.ModuleDict()
        for lc in self.levels_cfg:
            sup = lc.get("supervision") or {}
            if "cls" in sup:
                ccfg = sup["cls"]
                self.cls_heads[lc["name"]] = PerTokenClsHead(
                    dim=token_dim,
                    num_classes=int(ccfg["num_classes"]),
                    hidden_dim=int(ccfg.get("hidden_dim", 0)),
                    dropout=float(ccfg.get("dropout", 0.0)),
                )

        loss_kwargs = dict(loss_kwargs or {})
        loss_kwargs.setdefault("num_classes", num_classes)
        if not self.enable_origin_head:
            loss_kwargs["weight_origin"] = 0.0
        self.loss_fn = LArFormerLoss(levels_cfg=self.levels_cfg, **loss_kwargs)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _encode(self, batched_dict: dict) -> torch.Tensor:
        bb_in = {
            "coord":      batched_dict["coord_norm"],
            "feat":       batched_dict["feat"],
            "offset":     batched_dict["offset"],
            "grid_coord": batched_dict["grid_coord"],
        }
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(bb_in, return_point=True)
        else:
            result = self.backbone(bb_in, return_point=True)
        return result["point"].feat

    # ------------------------------------------------------------------
    # Per-event slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _per_event_slices(batched_dict: dict) -> List[dict]:
        offsets = batched_dict["offset"].detach().cpu().tolist()
        events = []
        prev = 0
        for ei, o in enumerate(offsets):
            events.append({
                "ei":       ei,
                "sp_slice": slice(prev, o),
                "n_sp":     o - prev,
            })
            prev = o
        return events

    def _build_event_dict(self, batched_dict: dict, ev: dict) -> dict:
        """Per-event dict consumed by LevelBuilder.forward + per_sp_labels.

        We pass everything that any builder might want; builders ignore keys
        they don't recognize.
        """
        sp = ev["sp_slice"]
        ed = {
            "coord_norm": batched_dict["coord_norm"][sp],
            "feat":       batched_dict["feat"][sp],
            "n_sp":       ev["n_sp"],
        }
        # Optional per-event/per-SP fields passed through unchanged when present.
        for k in ("strength", "wire"):
            if k in batched_dict:
                ed[k] = batched_dict[k][sp]
        # Fragment fields — list[B], pick this event's entry.
        for k in ("fragment_indices_per_event",
                  "fragment_trackid_per_event",
                  "fragment_pid_per_event",
                  "fragment_type_per_event"):
            if k in batched_dict:
                ed[k.replace("_per_event", "")] = batched_dict[k][ev["ei"]]
        return ed

    def _per_sp_labels_for_event(
        self, batched_dict: dict, ev: dict,
    ) -> Dict[str, torch.Tensor]:
        """Build the per_sp_labels dict the loss expects.

        We expose every per-spacepoint integer truth field present in the
        batch; the loss / cls heads pick by name through label_src.
        """
        sp = ev["sp_slice"]
        out: Dict[str, torch.Tensor] = {}
        for k in ("hasmatch", "origin_label", "ssnet_label",
                  "trackid", "pid", "slice_id"):
            if k in batched_dict:
                out[k] = batched_dict[k][sp]
        return out

    # ------------------------------------------------------------------
    # Public helpers (used by the GT visualizer in P4b)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_levels(
        self,
        data_dict: dict,
        zero_features: bool = True,
    ) -> List["OrderedDict[str, LevelOutput]"]:
        """Run every level builder without the backbone forward pass.

        With `zero_features=True`, the per-spacepoint feature tensor fed to
        the builders is all zeros — fine for GT visualization where we only
        need each level's `coords` and `sp_to_level_id`. The tokens are
        meaningless under this mode (set with care if a builder's projection
        is non-linear), but the geometry is exact.

        Returns: list[B] of OrderedDict[level_name → LevelOutput].
        """
        events = self._per_event_slices(data_dict)
        out: List["OrderedDict[str, LevelOutput]"] = []
        device = data_dict["coord_norm"].device
        for ev in events:
            sp = ev["sp_slice"]
            coord_norm = data_dict["coord_norm"][sp]
            if zero_features:
                sp_feat = torch.zeros(
                    ev["n_sp"], self.backbone_out_channels,
                    dtype=coord_norm.dtype, device=device,
                )
            else:
                # Caller must have already populated backbone features —
                # rare path, kept for completeness.
                sp_feat = data_dict["sp_feat"][sp]
            event_dict = self._build_event_dict(data_dict, ev)
            out.append(self.tokenizer(sp_feat, coord_norm, event_dict))
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict: dict) -> dict:
        sp_feat_all = self._encode(data_dict)
        events = self._per_event_slices(data_dict)

        per_event_loss = []
        per_event_pred = []
        for ev in events:
            sp = ev["sp_slice"]
            coord_norm = data_dict["coord_norm"][sp]
            sp_feat = sp_feat_all[sp]
            event_dict = self._build_event_dict(data_dict, ev)

            levels = self.tokenizer(sp_feat, coord_norm, event_dict)
            levels = self.token_refiner(levels)
            decoder_out = (self.decoder(levels)
                           if self.decoder is not None else None)

            # Per-level cls logits (one head per level that declared cls)
            per_level_cls: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for name, head in self.cls_heads.items():
                lvl = levels[name]
                if lvl.n_tokens == 0:
                    per_level_cls[name] = lvl.tokens.new_zeros(0, head.num_classes)
                else:
                    per_level_cls[name] = head(lvl.tokens)

            pred: dict = {
                "per_level_cls":   per_level_cls,
                # Level coords + sp_to_level_id are useful for downstream
                # analysis / the cascade Stage 1 → Stage 2 carry.
                "levels": {
                    name: {"coords": lvl.coords,
                           "sp_to_level_id": lvl.sp_to_level_id}
                    for name, lvl in levels.items()
                },
            }
            if decoder_out is not None:
                pred["class_logits"] = decoder_out["final"]["class_logits"]
                pred["origin"]       = decoder_out["final"]["origin"]
                pred["mask_logits"]  = decoder_out["final"]["mask_logits"]
            per_event_pred.append(pred)

            if self.training:
                gt_instances = data_dict["gt_instances_per_event"][ev["ei"]]
                per_sp_labels = self._per_sp_labels_for_event(data_dict, ev)
                loss_dict = self.loss_fn(
                    decoder_output=decoder_out,
                    levels=levels,
                    gt_instances=gt_instances,
                    per_sp_labels=per_sp_labels,
                    per_level_cls_logits=per_level_cls,
                )
                per_event_loss.append(loss_dict)
            elif "gt_instances_per_event" in data_dict:
                # Eval-with-GT path: compute matching + loss per event so
                # the evaluator can read per-pair quality metrics without
                # a second forward pass. The trainer's eval loop doesn't
                # need these, but the evaluator hook does.
                gt_instances = data_dict["gt_instances_per_event"][ev["ei"]]
                per_sp_labels = self._per_sp_labels_for_event(data_dict, ev)
                with torch.no_grad():
                    eval_loss = self.loss_fn(
                        decoder_output=decoder_out,
                        levels=levels,
                        gt_instances=gt_instances,
                        per_sp_labels=per_sp_labels,
                        per_level_cls_logits=per_level_cls,
                        return_matching=True,
                    )
                pred["eval_loss"] = eval_loss

        if self.training:
            # Pointcept's InformationWriter calls .item() on every dict value;
            # keep only 0-d tensors at the top level (no nested dicts/lists).
            # Union the keys (different events may have different non-zero
            # keys if some lack instances entirely).
            keys = set()
            for d in per_event_loss:
                keys |= set(d.keys())
            agg: Dict[str, torch.Tensor] = {}
            for k in keys:
                vals = [d[k].float() for d in per_event_loss if k in d]
                if vals:
                    agg[k] = torch.stack(vals).mean()
            out: Dict[str, torch.Tensor] = {"loss": agg.pop("total")}
            for k, v in agg.items():
                out[f"loss_{k}"] = v
            return out
        return {"predictions": per_event_pred}
