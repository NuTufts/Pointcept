"""
Top-level shower-clustering Mask2Former model. Phase 7 of the design
(see pointcept/docs/shower_clustering_design.md §3).

Wires four pieces together:
    1. (frozen) Sonata backbone        — per-spacepoint features
    2. ShowerClusteringTokenizer       — voxel + fragment + spacepoint tokens
    3. Mask2FormerDecoder              — N learnable queries, L decoder layers
    4. ShowerClusteringLoss            — Hungarian-matched set loss
                                         (deep supervision)

Forward semantics — Pointcept-style:

    out = model(batched_data_dict)
    if model.training:
        loss = out["loss"]; loss.backward()
    else:
        # out["predictions"] = list[B] of decoder output dicts

Batching: backbone runs once on the flat-batched (sum N_b, 6) input dict.
Tokenizer / decoder / loss are then applied per event because Hungarian
matching is per-event. Loss is averaged across events.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model

from .tokenizer import ShowerClusteringTokenizer
from .decoder import Mask2FormerDecoder, DEFAULT_SCALE_PATTERN
from .losses import ShowerClusteringLoss


@MODELS.register_module()
class ShowerClusteringMask2Former(nn.Module):
    """Mask2Former-style shower clustering model.

    Args:
        backbone: dict — Sonata backbone config (built via Pointcept's
            MODELS registry; see configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py).
        backbone_out_channels: D output of the backbone (1088 for v6 Sonata).
        token_dim: internal token dimension (default 256).
        num_queries: N learnable queries (default 64).
        num_classes: total class slots including no_object slot (last). For
            5 origin types pass 6.
        no_object_class_id: index of no_object slot (default num_classes-1).
        num_decoder_layers: must equal len(scale_pattern); validated.
        scale_pattern: cross-attn scale per layer; default
            (voxel, fragment, voxel, fragment, spacepoint, spacepoint).
        num_heads: attention heads (default 8).
        mlp_ratio: FFN expansion (default 4.0).
        voxel_size_cm: voxel grid size in cm for the voxel scale (default 5).
        coord_scale: detector→normalized coord scale factor (default 179.55).
        freeze_backbone: if True, no gradients through backbone (default
            True; matches V3 + the design doc).
        loss_kwargs: dict of overrides for ShowerClusteringLoss. Pass
            empty dict for defaults.
        frag_pool_*: passed through to ShowerClusteringTokenizer.
    """

    def __init__(
        self,
        backbone: dict,
        backbone_out_channels: int = 1088,
        token_dim: int = 256,
        num_queries: int = 64,
        num_classes: int = 6,
        no_object_class_id: Optional[int] = None,
        num_decoder_layers: int = 6,
        scale_pattern=DEFAULT_SCALE_PATTERN,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        voxel_size_cm: float = 5.0,
        coord_scale: float = 179.55,
        freeze_backbone: bool = True,
        frag_pool_layers: int = 2,
        frag_pool_heads: int = 8,
        frag_pool_max_points: int = 512,
        pos_emb_hidden_dim: Optional[int] = None,
        enable_origin_head: bool = True,
        loss_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone_out_channels = int(backbone_out_channels)
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.voxel_size_norm = float(voxel_size_cm) / float(coord_scale)

        self.tokenizer = ShowerClusteringTokenizer(
            in_dim=backbone_out_channels,
            token_dim=token_dim,
            voxel_size_norm=self.voxel_size_norm,
            frag_pool_layers=frag_pool_layers,
            frag_pool_heads=frag_pool_heads,
            frag_pool_max_points=frag_pool_max_points,
        )

        self.enable_origin_head = bool(enable_origin_head)
        self.decoder = Mask2FormerDecoder(
            dim=token_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_layers=num_decoder_layers,
            scale_pattern=scale_pattern,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_emb_hidden_dim=pos_emb_hidden_dim,
            enable_origin_head=self.enable_origin_head,
        )

        loss_kwargs = dict(loss_kwargs or {})
        loss_kwargs.setdefault("num_classes", num_classes)
        if no_object_class_id is not None:
            loss_kwargs.setdefault("no_object_class_id", no_object_class_id)
        if not self.enable_origin_head:
            # Origin head disabled → outputs are identically zero. Force the
            # loss/matcher weight to 0 regardless of what the user passed,
            # otherwise the matcher pays a constant origin cost on every
            # candidate pair (and wandb's loss_origin metric is meaningless).
            loss_kwargs["weight_origin"] = 0.0
        self.loss_fn = ShowerClusteringLoss(**loss_kwargs)

    # ------------------------------------------------------------------
    # Backbone wrapper
    # ------------------------------------------------------------------

    def encode_event(self, batched_dict: dict) -> torch.Tensor:
        """Run the (frozen) backbone on the flat-batched input and return
        per-spacepoint features (sum N_b, backbone_out_channels)."""
        bb_input = {
            "coord": batched_dict["coord_norm"],
            "feat": batched_dict["feat"],
            "offset": batched_dict["offset"],
            "grid_coord": batched_dict["grid_coord"],
        }
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(bb_input, return_point=True)
        else:
            result = self.backbone(bb_input, return_point=True)
        return result["point"].feat

    # ------------------------------------------------------------------
    # Per-event slicing
    # ------------------------------------------------------------------

    def _per_event_slices(self, batched_dict: dict) -> List[dict]:
        """Slice batched tensors back into per-event dicts."""
        offsets = batched_dict["offset"].detach().cpu().tolist()
        voxel_offsets = batched_dict["voxel_offset"].detach().cpu().tolist()
        events = []
        prev_o = 0
        prev_v = 0
        for ei in range(len(offsets)):
            n_sp = offsets[ei] - prev_o
            n_vox = voxel_offsets[ei] - prev_v
            events.append({
                "n_sp": n_sp,
                "n_vox": n_vox,
                "sp_slice": slice(prev_o, offsets[ei]),
                "vox_slice": slice(prev_v, voxel_offsets[ei]),
                "voxel_id_offset": prev_v,
                "fragment_indices": batched_dict["fragment_indices_per_event"][ei],
                "fragment_trackid": batched_dict["fragment_trackid_per_event"][ei],
                "gt_instances": batched_dict["gt_instances_per_event"][ei],
            })
            prev_o = offsets[ei]
            prev_v = voxel_offsets[ei]
        return events

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict: dict) -> dict:
        sp_feat_all = self.encode_event(data_dict)
        events = self._per_event_slices(data_dict)

        device = sp_feat_all.device
        per_event_loss = []
        per_event_pred = []
        for e in events:
            sp = e["sp_slice"]
            vx = e["vox_slice"]

            # Per-event tensors. The DataLoader's collate produces per-event
            # *lists* of CPU tensors for fragment fields; the trainer's
            # default `.cuda()` only moves top-level tensors so we move
            # these here defensively.
            sp_feat = sp_feat_all[sp]
            coord_norm = data_dict["coord_norm"][sp]
            strength = data_dict["feat"][sp, 3:6]
            voxel_id_local = data_dict["voxel_id"][sp] - e["voxel_id_offset"]
            voxel_keys = data_dict["voxel_keys"][vx]
            fragment_indices = [idx.to(device, non_blocking=True)
                                for idx in e["fragment_indices"]]
            fragment_trackid = e["fragment_trackid"].to(
                device, non_blocking=True)

            tokens = self.tokenizer(
                sp_feat=sp_feat,
                coord_norm=coord_norm,
                strength=strength,
                voxel_id=voxel_id_local,
                voxel_keys=voxel_keys,
                n_voxels=e["n_vox"],
                fragment_indices=fragment_indices,
            )
            decoder_out = self.decoder(
                voxel_tokens=tokens["voxel_tokens"],
                voxel_coords=tokens["voxel_coords"],
                fragment_tokens=tokens["fragment_tokens"],
                fragment_coords=tokens["fragment_coords"],
                spacepoint_tokens=tokens["spacepoint_tokens"],
                spacepoint_coords=tokens["spacepoint_coords"],
            )
            per_event_pred.append(decoder_out)

            if self.training:
                loss_dict = self.loss_fn(
                    decoder_output=decoder_out,
                    gt_instances=e["gt_instances"],
                    voxel_id=voxel_id_local,
                    n_voxels=e["n_vox"],
                    n_spacepoints=e["n_sp"],
                    fragment_trackid=fragment_trackid,
                )
                per_event_loss.append(loss_dict)

        if self.training:
            # Mean-aggregate per-component scalar losses across events.
            # Pointcept's InformationWriter calls `.item()` on every value
            # in the returned dict, so we must keep ONLY 0-d tensor values
            # at the top level — no nested dicts or lists.
            keys = per_event_loss[0].keys()
            agg = {}
            for k in keys:
                stacked = torch.stack([d[k].float() for d in per_event_loss])
                agg[k] = stacked.mean()
            # The "total" loss must be named "loss" for the trainer's backward.
            out = {"loss": agg.pop("total")}
            # Per-component scalars get logged as separate metrics. Prefix
            # to keep them grouped in tensorboard / stdout.
            for k, v in agg.items():
                out[f"loss_{k}"] = v
            return out
        else:
            return {"predictions": per_event_pred}
