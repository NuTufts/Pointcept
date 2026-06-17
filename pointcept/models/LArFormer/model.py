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
from pointcept.utils.logger import get_root_logger

from .builders import LevelOutput
from .decoder import Mask2FormerDecoder
from .heads import PerTokenClsHead
from .losses import LArFormerLoss
from .keypoint_heads import KeypointScoreHead, KeypointOffsetHead
from .keypoint_refine import KeypointRefinementDecoder
from .query_denoising import MaskDenoiser
from .query_selection import MixedQuerySelector
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
        mixed_query_selection: optional dict — when provided, build a
            MixedQuerySelector that picks Q anchor tokens from a source
            level (default voxel_8cm) and feeds them into the decoder as
            initial query content + positional anchor. The decoder's
            `query_content` / `query_pos` are then zero-initialized so
            they act as learnable per-slot deltas on top of the selected
            anchors. Source level must declare `supervision.cls` so the
            cls head can score tokens. See `query_selection.py`.
        mask_denoising: optional dict — when provided, build a
            MaskDenoiser that emits `dn_groups` denoising queries per
            GT instance (capped per event) and supervises them DIRECTLY
            (no Hungarian) against the original GT. Requires
            `mixed_query_selection` to also be enabled (DN queries are
            appended after the selector's K_reg regular queries; the
            decoder's learnable query_content / query_pos must be zero-
            init so they don't bleed into DN slots). Train-only —
            the eval path skips DN entirely. See `query_denoising.py`.
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
        unfreeze_decoder: bool = False,
        enable_origin_head: bool = True,
        loss_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        token_refiner: Optional[dict] = None,
        capture_decoder_stages: bool = False,
        mixed_query_selection: Optional[dict] = None,
        mask_denoising: Optional[dict] = None,
        ptv3_decoder_init_scale: float = 0.0,
        backbone_weight: Optional[str] = None,
        enable_keypoint_head: bool = False,
        keypoint_refine: Optional[dict] = None,
        enable_keypoint_dense_head: bool = False,
        keypoint_dense_cfg: Optional[dict] = None,
        freeze_non_keypoint: bool = False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if backbone_weight is not None:
            # Load before applying requires_grad — load_state_dict only
            # writes tensor data, not requires_grad. Frozen modules can
            # still be loaded.
            self._load_backbone_weight(backbone_weight)
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_decoder = bool(unfreeze_decoder)
        if self.freeze_backbone:
            for n, p in self.backbone.named_parameters():
                # When unfreeze_decoder is set, keep PT-v3m2 decoder
                # blocks trainable (`*.dec.*` in name). The encoder + every
                # other backbone module stays frozen.
                if self.unfreeze_decoder and (".dec." in n or n.endswith(".dec")
                                              or ".dec." in f".{n}"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.ptv3_decoder_init_scale = float(ptv3_decoder_init_scale)
        if self.unfreeze_decoder:
            # Initialize PT-v3m2 decoder Block weights. `cpe.Linear`
            # always goes to exactly zero (it's the actual NaN source
            # from sparse-conv edge cases). `attn.qkv.weight`,
            # `attn.proj.weight`, `mlp.fc2.weight` go to zero when
            # ptv3_decoder_init_scale == 0 (DETR-style residual identity
            # at init); otherwise truncated-normal with std=scale —
            # small-mag init lets gradient flow into the block from
            # iteration 1 while keeping the cpe NaN source guarded.
            # Plus a per-Block forward hook that sanitizes point.feat
            # for the attention-internals NaN cases the cpe zero doesn't
            # catch.
            self._init_ptv3_decoder_blocks(scale=self.ptv3_decoder_init_scale)
            self._register_ptv3_decoder_block_sanitizers()
        self.backbone_out_channels = int(backbone_out_channels)

        # Forward-hook capture of PT-v3m2 decoder stages (the
        # PTv3DecoderStageLevel builders read from this dict). Cleared
        # before each _encode() call; populated as the backbone forward
        # walks self.backbone.<teacher|student>.backbone.dec.dec{s}.
        self.capture_decoder_stages = bool(capture_decoder_stages)
        self._dec_stage_capture: dict = {}
        if self.capture_decoder_stages:
            self._register_decoder_stage_hooks()
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.enable_origin_head = bool(enable_origin_head)
        self.enable_keypoint_head = bool(enable_keypoint_head)
        if self.enable_keypoint_head and not self.enable_origin_head:
            raise ValueError(
                "enable_keypoint_head requires enable_origin_head "
                "(the origin head supplies the keypoint 'start')."
            )

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
                enable_keypoint_head=self.enable_keypoint_head,
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

        # Mixed query selection (Phase A): pick K anchor tokens from a
        # source level and feed them in as initial query content + anchor
        # coords. When active, the decoder's `query_content` / `query_pos`
        # parameters get zero-initialized (so they act as per-slot
        # learnable deltas on the selected anchors).
        self.query_selector: Optional[MixedQuerySelector] = None
        if mixed_query_selection is not None:
            if self.decoder is None:
                raise ValueError(
                    "mixed_query_selection requires num_queries > 0 "
                    "(a Mask2Former decoder must be built)."
                )
            qs_cfg = dict(mixed_query_selection)
            qs_cfg.setdefault("token_dim", token_dim)
            qs_cfg.setdefault("num_queries", num_queries)
            src = qs_cfg.get("source_level", "voxel_8cm")
            qs_cfg["source_level"] = src
            score_source = qs_cfg.get("score_source", "cls_head")
            if score_source == "cls_head" and src not in self.cls_heads:
                raise ValueError(
                    f"mixed_query_selection with score_source='cls_head' "
                    f"requires source_level={src!r} to declare "
                    f"supervision.cls in its level config so the per-level "
                    f"cls head exists. Levels with cls heads: "
                    f"{list(self.cls_heads.keys())!r}"
                )
            self.query_selector = MixedQuerySelector(**qs_cfg)
            # Zero the decoder's learnable query embeddings so they act
            # as deltas on top of the anchor features/coords.
            nn.init.zeros_(self.decoder.query_content)
            nn.init.zeros_(self.decoder.query_pos)

        loss_kwargs = dict(loss_kwargs or {})
        loss_kwargs.setdefault("num_classes", num_classes)
        if not self.enable_origin_head:
            loss_kwargs["weight_origin"] = 0.0
        loss_kwargs["enable_keypoint_head"] = self.enable_keypoint_head
        self.loss_fn = LArFormerLoss(levels_cfg=self.levels_cfg, **loss_kwargs)

        # Mask DINO-style denoising (Phase B). Train-time auxiliary path:
        # appends `dn_groups × n_gt` denoising queries after the regular
        # K_reg queries, supervised directly against their GT (no
        # Hungarian). Requires mixed_query_selection so that the decoder's
        # learnable `query_content` / `query_pos` are zero-initialized —
        # otherwise they'd leak into the DN slots' content.
        self.denoiser: Optional[MaskDenoiser] = None
        if mask_denoising is not None:
            if self.decoder is None:
                raise ValueError(
                    "mask_denoising requires num_queries > 0 (a Mask2Former "
                    "decoder must be built)."
                )
            if self.query_selector is None:
                raise ValueError(
                    "mask_denoising requires mixed_query_selection to be "
                    "enabled — the decoder's query_content / query_pos must "
                    "be zero-init for the regular path so they don't bleed "
                    "into the appended DN slots."
                )
            dn_cfg = dict(mask_denoising)
            dn_cfg.setdefault("token_dim", token_dim)
            dn_cfg.setdefault("num_classes", num_classes)
            self.denoiser = MaskDenoiser(**dn_cfg)

        # Phase-3 (2B) keypoint refinement decoder. Opt-in: requires the
        # keypoint head + a Mask2Former decoder. Runs after the main decoder,
        # cross-attending each query to its own spacepoints to sharpen
        # start/end (identity at init — composes with a trained 2A ckpt).
        self.keypoint_refine: Optional[KeypointRefinementDecoder] = None
        if keypoint_refine is not None:
            if not self.enable_keypoint_head:
                raise ValueError(
                    "keypoint_refine requires enable_keypoint_head=True.")
            if self.decoder is None:
                raise ValueError(
                    "keypoint_refine requires num_queries > 0 (a decoder).")
            kr_cfg = dict(keypoint_refine)
            kr_cfg.setdefault("dim", token_dim)
            self.keypoint_refine = KeypointRefinementDecoder(**kr_cfg)

        # Optional DENSE per-SP keypoint head (the Phase-1 PPN-style score +
        # offset heads) ON THE BACKBONE features — independent of the queries.
        # Lets ONE model emit both the per-particle query keypoints AND the
        # slice-level nu vertex (decoded from the dense score/offset field).
        # Supervised by the per-SP `kpscores` / `kpoffsets` the keypoint
        # datasets emit. Cheap; backbone can stay frozen (the dense field is
        # learnable from frozen features — see lartpc_data_prep/larformer_keypoint).
        self.enable_keypoint_dense_head = bool(enable_keypoint_dense_head)
        self.kp_dense_score_head = None
        self.kp_dense_offset_head = None
        if self.enable_keypoint_dense_head:
            kd = dict(keypoint_dense_cfg or {})
            self.kp_dense_n_types = int(kd.get("n_keypoint_types", 6))
            self.kp_dense_pos_weight = float(kd.get("pos_weight", 50.0))
            self.kp_dense_pos_threshold = float(kd.get("pos_threshold", 0.05))
            self.kp_dense_offset_sup_thresh = float(
                kd.get("offset_supervision_threshold", 0.05))
            self.kp_dense_w_score = float(kd.get("weight_score", 1.0))
            self.kp_dense_w_offset = float(kd.get("weight_offset", 1.0))
            self.kp_dense_coord_scale = float(kd.get("coord_scale", 179.55))
            hid = int(kd.get("head_hidden_dim", 256))
            # Opt-in coordinate pos-emb on the per-point dense heads (off by
            # default). Gives the MLP an explicit position handle on top of the
            # frozen backbone feature. `pos_emb` selects the kind
            # (None/"sinusoidal"/"mlp"); the rest are knobs for it.
            pe_kw = dict(
                pos_emb_kind=kd.get("pos_emb", None),
                pos_emb_dim=kd.get("pos_emb_dim", None),
                pos_emb_num_freq=kd.get("pos_emb_num_freq", None),
                pos_emb_max_freq=float(kd.get("pos_emb_max_freq", 256.0)),
                pos_emb_hidden_dim=kd.get("pos_emb_hidden_dim", None),
            )
            self.kp_dense_score_head = KeypointScoreHead(
                in_dim=backbone_out_channels, n_types=self.kp_dense_n_types,
                hidden_dim=hid, **pe_kw)
            if bool(kd.get("enable_offset_head", True)):
                self.kp_dense_offset_head = KeypointOffsetHead(
                    in_dim=backbone_out_channels, n_types=self.kp_dense_n_types,
                    hidden_dim=hid, **pe_kw)

        # Keypoint-only training: freeze the WHOLE particle-segmentation network
        # (backbone, PT-v3 decoder, token refiner, Mask2Former decoder layers +
        # class/mask heads, per-level cls heads, query selection, denoising) and
        # train ONLY the keypoint-specific params — the per-query keypoint heads
        # (start=origin / end / uncertainty / end-gate), the 2B refinement
        # decoder, and the dense head. Preserves the loaded Stage-3 quality and
        # directs all optimization to keypoints. Applied LAST so it overrides
        # freeze_backbone / unfreeze_decoder. Pair with unfreeze_decoder=False
        # so `_encode` runs the backbone under no_grad (no wasted graph).
        self.freeze_non_keypoint = bool(freeze_non_keypoint)
        if self.freeze_non_keypoint:
            # The decoder needs its fp32-safe attention backend set up. If
            # unfreeze_decoder already did it (_init_ptv3_decoder_blocks),
            # skip; otherwise do the forward-only part here so a frozen
            # decoder still produces correct masks (IoU regression fix).
            if not self.unfreeze_decoder:
                self._ensure_decoder_fp32_forward()
            self._apply_keypoint_only_freeze()

        # Defense-in-depth: register a NaN/Inf-zeroing hook on every
        # trainable parameter's gradient. This guarantees no NaN/Inf
        # reaches the optimizer regardless of which internal layer
        # produced it (LayerNorm of an Inf input, softmax over an Inf-
        # scale Q@K, etc.). Critical for the early iters of configs
        # where multiple from-scratch subsystems (PTv3 decoder + token
        # refiner + Mask2Former decoder) co-train. `clip_grad_norm_`
        # alone is NOT enough — it computes total_norm = sqrt(sum(g²)),
        # which is NaN if any single grad has NaN, leaving every
        # gradient un-clipped. This hook fires BEFORE clip_grad_norm_,
        # so by the time the trainer clips, the gradients are at worst
        # finite-but-large (which clip_grad_norm_ then bounds).
        self._register_nan_safe_grad_hooks()

    @staticmethod
    def _nan_safe_grad_hook(g: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # Substrings that identify the KEYPOINT-specific parameters (everything
    # else — class/mask heads, decoder layers, refiner, backbone — is the
    # particle-segmentation network). `.origin_head.` is the start point.
    _KEYPOINT_PARAM_MARKERS = (
        "keypoint_refine", "kp_dense_score_head", "kp_dense_offset_head",
        ".origin_head.", ".end_head.", ".start_logvar_head.",
        ".end_logvar_head.", ".end_exist_head.", "kp_pos_emb",
    )

    def _ensure_decoder_fp32_forward(self) -> None:
        """Set the PT-v3 decoder blocks' attention to the xformers backend so
        the fp32 eval/inference forward is correct.

        The PT-v3 native decoder (enc_mode=False) is normally set up by
        `_init_ptv3_decoder_blocks` — but that only runs under
        `unfreeze_decoder=True`. With the decoder frozen
        (freeze_non_keypoint + unfreeze_decoder=False) that setup is skipped,
        and the blocks keep `flash_attn`, which does NOT handle fp32 and
        silently produces garbage masks (IoU→0). This applies just the
        forward-relevant part (backend + NaN sanitizers), WITHOUT touching
        weights — the loaded checkpoint's decoder weights are preserved.
        No-op on backbones without a decoder."""
        ptv3 = self._find_ptv3_inner()
        dec = getattr(ptv3, "dec", None)
        if dec is None:
            return
        n = 0
        for _sn, stage_seq in dec.named_children():
            for child_name, mod in stage_seq.named_children():
                if not child_name.startswith("block"):
                    continue
                attn = getattr(mod, "attn", None)
                if attn is not None and hasattr(attn, "flash_backend"):
                    attn.flash_backend = "xformers"
                    n += 1
        self._register_ptv3_decoder_block_sanitizers()
        get_root_logger().info(
            f"[LArFormer] freeze_non_keypoint: set xformers backend on {n} "
            f"PT-v3 decoder blocks (fp32-safe forward).")

    def _apply_keypoint_only_freeze(self) -> None:
        """requires_grad=True ONLY for keypoint params; freeze everything else."""
        n_train = n_freeze = 0
        for name, p in self.named_parameters():
            keep = any(m in name for m in self._KEYPOINT_PARAM_MARKERS)
            p.requires_grad = bool(keep)
            if keep:
                n_train += 1
            else:
                n_freeze += 1
        get_root_logger().info(
            f"[LArFormer] freeze_non_keypoint: {n_train} keypoint param-"
            f"tensors trainable, {n_freeze} frozen (particle network).")

    def _load_backbone_weight(self, weight_path: str) -> None:
        """Load a Sonata pretrain (or similar) into self.backbone.

        Uses the same prefix-stripping + shape-aware filter as the cascade
        wrappers (CascadedSlicer._load_state_dict_into). Lets a standalone
        LArFormer config that doesn't sit under a cascade still pick up
        the trained backbone weights via:

            model = dict(
                type="LArFormer",
                backbone=...,
                backbone_weight="path/to/sonata_pretrain.pth",
                ...,
            )

        Mismatched keys (shape conflict between ckpt and current
        backbone — typically the `mask_token` slots that depend on
        `mask_token=True` in the PT-v3 config) are dropped with a
        printed count rather than raising.
        """
        ckpt = torch.load(weight_path, map_location="cpu",
                          weights_only=False)
        if "state_dict" not in ckpt:
            raise KeyError(
                f"{weight_path}: no 'state_dict' key; got "
                f"{list(ckpt.keys())[:10]}"
            )
        sd = ckpt["state_dict"]
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        if sd and all(k.startswith("backbone.") for k in sd.keys()):
            sd = {k[len("backbone."):]: v for k, v in sd.items()}
        target_shapes = {k: tuple(v.shape) for k, v
                         in self.backbone.state_dict().items()}
        filtered_sd: dict = {}
        n_shape_mismatch = 0
        first_mismatches: list = []
        for k, v in sd.items():
            if k not in target_shapes:
                filtered_sd[k] = v
                continue
            if tuple(v.shape) == target_shapes[k]:
                filtered_sd[k] = v
            else:
                n_shape_mismatch += 1
                if len(first_mismatches) < 5:
                    first_mismatches.append(
                        f"{k}: ckpt {tuple(v.shape)} vs model {target_shapes[k]}"
                    )
        missing, unexpected = self.backbone.load_state_dict(
            filtered_sd, strict=False,
        )
        print(f"[LArFormer] Loaded backbone weight: {weight_path}")
        print(f"  missing={len(missing)}  unexpected={len(unexpected)}  "
              f"shape_mismatch_dropped={n_shape_mismatch}")
        if missing[:5]:
            print(f"  first missing: {missing[:5]}")
        if unexpected[:5]:
            print(f"  first unexpected: {unexpected[:5]}")
        if first_mismatches:
            print(f"  first shape-mismatch: {first_mismatches}")

    @staticmethod
    def _slice_decoder_output(decoder_out: dict, start: int, end: int) -> dict:
        """Slice each per-layer Q dimension of a decoder output to [start, end).

        Used to split a combined [regular | DN] decoder output back into
        independent regular / DN dicts that the loss + downstream
        evaluator can consume separately. Cheap — all slices share
        storage with the source tensors.
        """
        def _slice_layer(layer: dict) -> dict:
            out = {
                "class_logits": layer["class_logits"][start:end],
                "origin":       layer["origin"][start:end],
                "mask_logits":  OrderedDict(
                    (name, m[start:end])
                    for name, m in layer["mask_logits"].items()
                ),
            }
            if "query_embed" in layer:
                out["query_embed"] = layer["query_embed"][start:end]
            for k in ("end", "start_logvar", "end_logvar", "end_logit"):
                if k in layer:
                    out[k] = layer[k][start:end]
            if "scale" in layer:
                out["scale"] = layer["scale"]
            return out
        return {
            "init":   _slice_layer(decoder_out["init"]),
            "layers": [_slice_layer(l) for l in decoder_out["layers"]],
            "final":  _slice_layer(decoder_out["final"]),
        }

    def _register_nan_safe_grad_hooks(self) -> None:
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(self._nan_safe_grad_hook)

    def _init_ptv3_decoder_blocks(self, scale: float = 0.0) -> None:
        """Initialize weights of every Block inside the PT-v3m2 decoder.

        Two regimes, controlled by `scale`:

          - `scale == 0` (default): zero-init `attn.qkv.weight`,
            `attn.proj.weight`, `mlp.fc2.weight`. Each residual sub-
            update is identity at init → the Block passes features
            through unchanged → the from-scratch decoder's output
            equals the encoder's skip features on the first iter.
            Maximally stable; symmetry-breaking is slow because
            gradient into the inner Q/K/V weights only flows through
            the zeroed out_projs.

          - `scale > 0`: truncated-normal with std=scale on those same
            three weight matrices. The block is no longer identity at
            init, but gradients flow into the inner weights immediately
            so the block starts learning from iteration 1. Recommended
            value: ~0.01 (≈ 5–6× smaller than PyTorch's default Linear
            init for D=256, small enough to keep activations bounded
            but large enough to break symmetry).

        Unconditional in both regimes:
          - `cpe.Linear` (CPE output projection after SubMConv3d):
            always zero. This is the actual NaN source — the random-
            init sparse-conv → Linear chain produces Inf entries on
            edge-case point positions, and zeroing the Linear after
            the sparse conv neutralizes it. NEVER use scale > 0 here.
          - Biases on `cpe.Linear`, `attn.proj`, `mlp.fc2`: zero.
            `attn.qkv.bias` is kept at PyTorch's default Linear init
            (small uniform) so Q != K != V at scale=0 too — avoids
            the degenerate flash_attn backward NaN.
          - `attn.flash_backend = "xformers"`: switched on regardless
            of scale. flash_attn's bfloat16 backward NaN was the issue
            that originally forced this; xformers is comparably fast
            at the decoder's token counts (~5K per stage) and removes
            a risk surface.

        Only called when `unfreeze_decoder=True`. No-ops on backbones
        without a decoder attribute.
        """
        ptv3 = self._find_ptv3_inner()
        dec = getattr(ptv3, "dec", None)
        if dec is None:
            return
        import torch.nn as nn
        scale = float(scale)
        n_inited = 0
        for stage_name, stage_seq in dec.named_children():
            for child_name, mod in stage_seq.named_children():
                if not child_name.startswith("block"):
                    continue
                # cpe.Linear: ALWAYS zero (NaN source — see docstring).
                cpe_seq = getattr(mod, "cpe", None)
                if cpe_seq is not None:
                    for m in cpe_seq.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.zeros_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                            n_inited += 1
                            break  # only the first Linear (CPE output proj)
                # attn.qkv.weight / attn.proj.weight: zero (scale=0) or
                # trunc-normal(std=scale). Biases follow the rule in
                # the docstring.
                attn = getattr(mod, "attn", None)
                if attn is not None:
                    if hasattr(attn, "flash_backend"):
                        attn.flash_backend = "xformers"
                    qkv = getattr(attn, "qkv", None)
                    if isinstance(qkv, nn.Linear):
                        if scale == 0.0:
                            nn.init.zeros_(qkv.weight)
                        else:
                            nn.init.trunc_normal_(qkv.weight, std=scale)
                        n_inited += 1
                    proj = getattr(attn, "proj", None)
                    if isinstance(proj, nn.Linear):
                        if scale == 0.0:
                            nn.init.zeros_(proj.weight)
                        else:
                            nn.init.trunc_normal_(proj.weight, std=scale)
                        if proj.bias is not None:
                            nn.init.zeros_(proj.bias)
                        n_inited += 1
                # mlp.fc2.weight: same regime as the attn weights.
                mlp_seq = getattr(mod, "mlp", None)
                if mlp_seq is not None:
                    for m in mlp_seq.modules():
                        fc2 = getattr(m, "fc2", None)
                        if isinstance(fc2, nn.Linear):
                            if scale == 0.0:
                                nn.init.zeros_(fc2.weight)
                            else:
                                nn.init.trunc_normal_(fc2.weight, std=scale)
                            if fc2.bias is not None:
                                nn.init.zeros_(fc2.bias)
                            n_inited += 1
                            break
        self._n_ptv3_dec_blocks_inited = n_inited

    def _register_ptv3_decoder_block_sanitizers(self) -> None:
        """Forward-hook each PT-v3m2 decoder Block to sanitize point.feat
        on the way out. Belt-and-suspenders for cases where attention
        internals (e.g., flash attention) produce NaN that the zero-init
        on `attn.proj` doesn't catch because `0 * NaN = NaN`."""
        ptv3 = self._find_ptv3_inner()
        dec = getattr(ptv3, "dec", None)
        if dec is None:
            return

        def _hook(_mod, _inp, out):
            # Out is a Point-like object with .feat (and possibly
            # .sparse_conv_feat). Sanitize both in-place so any
            # downstream code that reads sparse_conv_feat doesn't see
            # NaN either.
            if hasattr(out, "feat"):
                out.feat = torch.nan_to_num(
                    out.feat, nan=0.0, posinf=1000.0, neginf=-1000.0,
                ).clamp(-1000.0, 1000.0)
                if hasattr(out, "sparse_conv_feat") and out.sparse_conv_feat is not None:
                    out.sparse_conv_feat = out.sparse_conv_feat.replace_feature(out.feat)
            return out

        n = 0
        for stage_name, stage_seq in dec.named_children():
            for child_name, mod in stage_seq.named_children():
                if child_name.startswith("block"):
                    mod.register_forward_hook(_hook)
                    n += 1
        self._n_ptv3_dec_block_sanitizer_hooks = n

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _find_ptv3_inner(self):
        """Locate the inner PT-v3m2 module so we can hook its decoder stages.

        Sonata-v1m1 wraps PT-v3m2 inside `self.teacher.backbone` (and a
        twin in `self.student.backbone`). Sonata-v1m1.forward(return_point=
        True) only runs teacher, so we hook teacher.backbone. For a bare
        PT-v3m2 backbone we just hook `self.backbone` directly.
        """
        bb = self.backbone
        if hasattr(bb, "teacher") and hasattr(bb.teacher, "backbone"):
            return bb.teacher.backbone
        if hasattr(bb, "student") and hasattr(bb.student, "backbone"):
            return bb.student.backbone
        return bb

    def _register_decoder_stage_hooks(self) -> None:
        """Register forward hooks on PT-v3m2's dec{s} submodules so each
        stage's output Point is captured into self._dec_stage_capture."""
        ptv3 = self._find_ptv3_inner()
        if getattr(ptv3, "enc_mode", True):
            raise ValueError(
                "capture_decoder_stages=True requires the inner PT-v3m2 "
                "backbone to be constructed with enc_mode=False. Update "
                "the backbone subconfig (and set up_cast_level=0 on the "
                "Sonata-v1m1 wrapper so it doesn't try to upcast non-"
                "existent pooling_parent chains)."
            )
        dec = getattr(ptv3, "dec", None)
        if dec is None:
            raise ValueError(
                "PT-v3m2 backbone has no self.dec — enc_mode is False but "
                "the decoder wasn't built. Check dec_depths / dec_channels "
                "etc. in the backbone config."
            )

        def _make_hook(name):
            def hook(_mod, _inp, output):
                self._dec_stage_capture[name] = output
            return hook

        # PointSequential stores its named children in self.dec._modules
        for child_name, child_mod in dec.named_children():
            if child_name.startswith("dec"):
                child_mod.register_forward_hook(_make_hook(child_name))

    def _encode(self, batched_dict: dict) -> torch.Tensor:
        bb_in = {
            "coord":      batched_dict["coord_norm"],
            "feat":       batched_dict["feat"],
            "offset":     batched_dict["offset"],
            "grid_coord": batched_dict["grid_coord"],
        }
        # Clear last forward's captured decoder stages BEFORE re-running
        # the backbone, so a hook miss surfaces as a clean KeyError later
        # rather than silently reusing stale values.
        if self.capture_decoder_stages:
            self._dec_stage_capture.clear()

        # If the decoder is trainable, we must NOT wrap the backbone in
        # no_grad — otherwise nothing flowing through the decoder builds
        # a grad-tracking graph. The encoder's params have
        # requires_grad=False, so they still won't accumulate gradients
        # even though their activations are differentiable; only the
        # decoder's params get optimizer updates.
        # freeze_non_keypoint freezes every backbone/decoder param, so the
        # backbone forward can always run under no_grad (no param needs its
        # graph), even if unfreeze_decoder was left True.
        use_no_grad = ((self.freeze_backbone and not self.unfreeze_decoder)
                       or getattr(self, "freeze_non_keypoint", False))
        if use_no_grad:
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
        # `particle_class_id` is the per-SP particle-class label produced
        # by `LArFormerStage12CacheDataset` (= dataset emits 0..5 for
        # visible particles, -1 = no GT). Available only on caches that
        # have been processed by
        # `tools/augment_stage12_cache_particle_class_id.py`.
        for k in ("hasmatch", "origin_label", "ssnet_label",
                  "trackid", "pid", "slice_id", "particle_class_id"):
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
    # PT-v3m2 decoder-stage per-event extractor
    # ------------------------------------------------------------------

    def _build_decoder_stages_per_event(
        self, data_dict: dict, events: List[dict],
    ) -> Optional[dict]:
        """Slice the hooked decoder-stage Points into per-event tensors and
        derive per-event sp_to_level_id by chaining pooling_inverse.

        Returns: {stage_name → list[B] of {tokens, coords, sp_to_level_id}}
        Or None if no stages were captured.

        Algorithm for sp_to_level_id at stage K:
            stage_1.pooling_inverse: (N_sp,)  → [0, M_1)
            stage_2.pooling_inverse: (M_1,)   → [0, M_2)
            stage_3.pooling_inverse: (M_2,)   → [0, M_3)
            sp_to_stage_K = stage_K.inv[…[stage_2.inv[stage_1.inv]]…]

        Per-event slicing: each stage's Point.batch is sorted by batch id
        (encoder's GridPooling packs batch into the high bits of grid_coord
        before unique), so we can use searchsorted on `batch` to find each
        event's contiguous token block.
        """
        if not self.capture_decoder_stages or not self._dec_stage_capture:
            return None

        # PT-v3m2 names its decoder stages dec0, dec1, dec2, dec3 (for the
        # standard 4-stride encoder). dec_s outputs a Point at stride 2^s.
        # Sort finest-first so the pooling_inverse chain composes correctly.
        def _stage_idx(name):
            return int(name[3:])

        ordered = sorted(self._dec_stage_capture.keys(), key=_stage_idx)
        # Skip dec0 (per-SP, same resolution as the spacepoint level): the
        # SpacepointBuilder covers that. A builder that wants dec0
        # specifically can be added later.
        ordered = [n for n in ordered if _stage_idx(n) >= 1]
        if not ordered:
            return None

        # Build the global sp→stage_K chain (sp index = stage_0 token id).
        sp_to_global: dict = {}
        cur = None
        for name in ordered:
            inv = self._dec_stage_capture[name].pooling_inverse
            if cur is None:
                cur = inv                          # (N_sp,) → [0, M_1)
            else:
                cur = inv[cur]                     # (N_sp,) → [0, M_k)
            sp_to_global[name] = cur

        # Per-event SP block boundaries from the dataset's offset.
        out: dict = {name: [] for name in ordered}
        for name in ordered:
            pt = self._dec_stage_capture[name]
            batch = pt.batch                                # (M_total,) sorted
            n_total = int(pt.feat.shape[0])
            # Per-event start offsets in this stage's token list.
            # searchsorted on sorted batch yields the first index >= ei.
            stage_starts = []
            for ev in events:
                ei = ev["ei"]
                idx = torch.searchsorted(
                    batch, torch.tensor(ei, device=batch.device),
                ).item()
                stage_starts.append(idx)
            stage_starts.append(n_total)

            sp_to_stage_global = sp_to_global[name]
            for ev in events:
                ei = ev["ei"]
                sp_slice = ev["sp_slice"]
                s = stage_starts[ei]
                e = stage_starts[ei + 1]
                # Sanitize stage features at the source. From-scratch
                # PT-v3m2 decoder Blocks can produce NaN/Inf features at
                # init; PTv3DecoderStageLevel projects these via Linear,
                # whose weight grad = stage_features.T @ grad_output goes
                # NaN if the input is NaN — same class of bug we hit on
                # the per-level cls heads.
                ev_tokens = torch.nan_to_num(
                    pt.feat[s:e], nan=0.0, posinf=1000.0, neginf=-1000.0,
                ).clamp(-1000.0, 1000.0)
                ev_coords = pt.coord[s:e]                   # (M_ev, 3)
                # Per-event sp_to_level_id in event-local indices [0, M_ev).
                ev_sp_global = sp_to_stage_global[sp_slice]  # (N_sp_ev,)
                ev_sp_to_level = (ev_sp_global - s).to(torch.long)
                out[name].append({
                    "tokens": ev_tokens,
                    "coords": ev_coords,
                    "sp_to_level_id": ev_sp_to_level,
                })
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _dense_keypoint_loss(self, logits, offsets, target_scores,
                             target_offsets):
        """Dense per-SP keypoint loss: weighted-MSE on sigmoid(score logits)
        vs `kpscores` + smooth-L1 offset vote (where score>thr) vs `kpoffsets`.
        Returns (grad_total, diag) with 0-d diag scalars. Mirrors
        LArFormerKeypoint's score/offset losses."""
        import torch.nn.functional as F
        target_scores = target_scores.to(logits.dtype)
        pred_s = torch.sigmoid(logits)
        se = (pred_s - target_scores) ** 2
        if self.kp_dense_pos_weight != 1.0:
            w = torch.where(target_scores > self.kp_dense_pos_threshold,
                            torch.full_like(target_scores, self.kp_dense_pos_weight),
                            torch.ones_like(target_scores))
            score_loss = (se * w).sum() / w.sum().clamp(min=1.0)
        else:
            score_loss = se.mean()
        total = self.kp_dense_w_score * score_loss
        diag = {"score": score_loss.detach()}
        with torch.no_grad():
            pos = target_scores > self.kp_dense_pos_threshold
            diag["mse_pos"] = (se[pos].mean() if pos.any()
                               else logits.new_zeros(()))
        if offsets is not None and target_offsets is not None:
            tgt_off = target_offsets.to(offsets.dtype)
            mask = target_scores > self.kp_dense_offset_sup_thresh   # (N, T)
            if bool(mask.any()):
                off_loss = F.smooth_l1_loss(offsets[mask], tgt_off[mask],
                                            reduction="mean")
                total = total + self.kp_dense_w_offset * off_loss
                diag["off"] = off_loss.detach()
                with torch.no_grad():
                    diag["off_err_cm"] = ((offsets[mask] - tgt_off[mask])
                                          .norm(dim=-1).mean()
                                          * self.kp_dense_coord_scale)
        return total, diag

    def forward(self, data_dict: dict) -> dict:
        sp_feat_all = self._encode(data_dict)
        # Sanitize backbone output at the source. Covers any from-scratch
        # PT-v3m2 decoder (when enc_mode=False + unfreeze_decoder=True)
        # whose random-init Blocks can produce NaN/Inf per-SP features
        # in the first iters. With clean sp_feat_all, the tokenizer's
        # VoxelBuilder / SpacepointBuilder / PTv3DecoderStageLevel all
        # produce clean tokens, and the refiner blocks' LayerNorm /
        # attention / FFN see in-range inputs (so their BACKWARD weight
        # grads — which depend on input.T @ grad_output — stay finite
        # even though their forward output is zeroed at init).
        # _INTERMEDIATE_CAP=1000 matches the decoder's intermediate
        # sanitizer convention (generous range; sigmoid/softmax don't
        # come anywhere near).
        sp_feat_all = torch.nan_to_num(
            sp_feat_all, nan=0.0, posinf=1000.0, neginf=-1000.0,
        ).clamp(-1000.0, 1000.0)
        events = self._per_event_slices(data_dict)
        per_event_dec_stages = self._build_decoder_stages_per_event(
            data_dict, events,
        )

        # Dense per-SP keypoint head (whole batch at once; sliced per event).
        # Score logits (N, n_types) + offset votes (N, n_types, 3).
        dense_logits_all = None
        dense_off_all = None
        if self.enable_keypoint_dense_head:
            # coord_norm aligns row-for-row with sp_feat_all (whole batch); the
            # heads use it only when they were built with a pos-emb.
            dense_coord_all = data_dict["coord_norm"]
            dense_logits_all = self.kp_dense_score_head(
                sp_feat_all, dense_coord_all)
            if self.kp_dense_offset_head is not None:
                dense_off_all = self.kp_dense_offset_head(
                    sp_feat_all, dense_coord_all)

        per_event_loss = []
        per_event_pred = []
        for ev in events:
            sp = ev["sp_slice"]
            coord_norm = data_dict["coord_norm"][sp]
            sp_feat = sp_feat_all[sp]
            event_dict = self._build_event_dict(data_dict, ev)
            if per_event_dec_stages is not None:
                event_dict["ptv3_dec_stages"] = {
                    name: per_event_dec_stages[name][ev["ei"]]
                    for name in per_event_dec_stages
                }

            levels = self.tokenizer(sp_feat, coord_norm, event_dict)
            levels = self.token_refiner(levels)
            # Sanitize tokens at the refiner output boundary. Both the
            # per-level cls heads AND the decoder consume `levels[name].
            # tokens`. If the refiner (random-init transformer blocks)
            # produces NaN/Inf in tokens, `head(tokens)` is clean in
            # forward but `∂(matmul)/∂(head.weight) = tokens.T` is NaN
            # in backward — corrupts the cls-head weight gradient.
            # Sanitizing once here covers both consumers.
            from .builders import LevelOutput as _LO
            from collections import OrderedDict as _OD
            levels = _OD(
                (name, _LO(
                    tokens=(torch.nan_to_num(
                        lvl.tokens, nan=0.0, posinf=1000.0, neginf=-1000.0,
                    ).clamp(-1000.0, 1000.0)
                            if lvl.tokens.shape[0] > 0 else lvl.tokens),
                    coords=lvl.coords,
                    sp_to_level_id=lvl.sp_to_level_id,
                    name=lvl.name,
                ))
                for name, lvl in levels.items()
            )
            # Per-level cls logits (one head per level that declared cls).
            # Sanitize the same way the decoder sanitizes its own predictions
            # (clamp + nan_to_num) — protects all downstream consumers (the
            # loss, the matcher, the evaluator) from pathological logits
            # while the deep freshly-random stack is warming up. Same
            # rationale as `Mask2FormerDecoder._sanitize_logits`.
            #
            # Computed BEFORE the decoder so MixedQuerySelector (when
            # active) can score source-level tokens via `1 - p(no_object)`
            # from this dict.
            per_level_cls: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for name, head in self.cls_heads.items():
                lvl = levels[name]
                if lvl.n_tokens == 0:
                    per_level_cls[name] = lvl.tokens.new_zeros(0, head.num_classes)
                else:
                    raw = head(lvl.tokens)
                    per_level_cls[name] = torch.nan_to_num(
                        raw.clamp(-50.0, 50.0),
                        nan=0.0, posinf=50.0, neginf=-50.0,
                    )

            # Denoising queries (Mask DINO Phase B): train-only path that
            # appends `dn_groups × n_gt` queries after the regular K_reg.
            # Eval / inference and the cls-only (decoder=None) path skip
            # this entirely. dn_init.n_queries == 0 (empty event) is
            # handled inside the decoder + loss as a no-op.
            dn_init = None
            dn_self_attn_mask = None
            if (self.decoder is not None and self.training
                    and self.denoiser is not None):
                gt_for_dn = data_dict["gt_instances_per_event"][ev["ei"]]
                dn_init = self.denoiser(
                    gt_instances=gt_for_dn,
                    device=sp_feat.device,
                    dtype=sp_feat.dtype,
                )
                if dn_init.n_queries > 0:
                    dn_self_attn_mask = MaskDenoiser.build_self_attn_mask(
                        n_regular=self.num_queries,
                        group_id=dn_init.group_id,
                    )

            if self.decoder is not None:
                if self.query_selector is not None:
                    init_q, init_anchor = self.query_selector(
                        levels, per_level_cls,
                    )
                    if dn_init is not None and dn_init.n_queries > 0:
                        init_q = torch.cat([init_q, dn_init.init_q], dim=0)
                        init_anchor = torch.cat(
                            [init_anchor, dn_init.init_anchor], dim=0,
                        )
                    decoder_out = self.decoder(
                        levels,
                        init_query_content=init_q,
                        init_anchor_coords=init_anchor,
                        dn_self_attn_mask=dn_self_attn_mask,
                    )
                else:
                    decoder_out = self.decoder(levels)
            else:
                decoder_out = None

            # Split decoder output into [regular | DN]. The regular slice
            # is what feeds the user-facing `pred` dict and the Hungarian
            # loss path; the DN slice goes to `compute_dn_loss`. With no
            # DN active (eval or denoiser=None), regular_dec is just
            # decoder_out and dn_dec is None.
            regular_dec = decoder_out
            dn_dec = None
            if (decoder_out is not None and dn_init is not None
                    and dn_init.n_queries > 0):
                regular_dec = self._slice_decoder_output(
                    decoder_out, 0, self.num_queries,
                )
                dn_dec = self._slice_decoder_output(
                    decoder_out, self.num_queries,
                    self.num_queries + dn_init.n_queries,
                )

            # Phase-3 (2B) keypoint refinement on the regular queries — each
            # query cross-attends to its own primary-level (spacepoint) tokens
            # to sharpen start/end. Returns per-layer dicts; the last layer is
            # the final keypoint prediction. None when 2B is off.
            refine_layers = None
            if (self.keypoint_refine is not None and regular_dec is not None
                    and "end" in regular_dec["final"]):
                fin = regular_dec["final"]
                prim = self.loss_fn.primary_level or "spacepoint"
                if prim in levels and prim in fin["mask_logits"]:
                    plvl = levels[prim]
                    refine_layers = self.keypoint_refine(
                        query_embed=fin["query_embed"],
                        init_start=fin["origin"], init_end=fin["end"],
                        init_end_logit=fin["end_logit"],
                        sp_tokens=plvl.tokens, sp_coords=plvl.coords,
                        sp_mask_logits=fin["mask_logits"][prim],
                    )

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
            if regular_dec is not None:
                pred["class_logits"] = regular_dec["final"]["class_logits"]
                pred["origin"]       = regular_dec["final"]["origin"]
                pred["mask_logits"]  = regular_dec["final"]["mask_logits"]
                # Per-query final embedding (Q, D) for the Phase-2 keypoint
                # head reading a (possibly frozen) Stage-3 LArFormer.
                if "query_embed" in regular_dec["final"]:
                    pred["query_embed"] = regular_dec["final"]["query_embed"]
                # Phase-2 keypoint predictions (start = `origin`). Present
                # only when enable_keypoint_head. `end_logit` sigmoids to the
                # track-end existence prob; logvars are per-axis.
                for k in ("end", "start_logvar", "end_logvar", "end_logit"):
                    if k in regular_dec["final"]:
                        pred[f"kp_{k}"] = regular_dec["final"][k]
                # Phase-3: the refined final layer supersedes the 2A keypoint
                # prediction (start = origin, end, uncertainty, gate).
                if refine_layers is not None:
                    rf = refine_layers[-1]
                    pred["origin"]          = rf["origin"]
                    pred["kp_end"]          = rf["end"]
                    pred["kp_start_logvar"] = rf["start_logvar"]
                    pred["kp_end_logvar"]   = rf["end_logvar"]
                    pred["kp_end_logit"]    = rf["end_logit"]
            # Dense per-SP keypoint field (score + offset votes), detector-frame
            # decoding happens downstream (evaluator / inference). The nu vertex
            # is decoded from this. coord_norm is this event's SP positions.
            if dense_logits_all is not None:
                pred["dense_kpscores"] = torch.sigmoid(
                    dense_logits_all[sp]).detach()
                pred["dense_coord_norm"] = coord_norm.detach()
                if dense_off_all is not None:
                    pred["dense_kpoffsets"] = dense_off_all[sp].detach()
            per_event_pred.append(pred)

            if self.training:
                gt_instances = data_dict["gt_instances_per_event"][ev["ei"]]
                per_sp_labels = self._per_sp_labels_for_event(data_dict, ev)
                # When DN is active we need the loss to also return GT
                # tensors (gt_classes / gt_origin / per_level_gt_mask) so
                # compute_dn_loss can reuse them without recomputing.
                want_dn = dn_dec is not None
                loss_dict = self.loss_fn(
                    decoder_output=regular_dec,
                    levels=levels,
                    gt_instances=gt_instances,
                    per_sp_labels=per_sp_labels,
                    per_level_cls_logits=per_level_cls,
                    return_matching=want_dn,
                    keypoint_refine_layers=refine_layers,
                )
                if want_dn:
                    dn_dict = self.loss_fn.compute_dn_loss(
                        decoder_output_dn=dn_dec,
                        per_level_gt_mask=loss_dict["per_level_gt_mask"],
                        gt_classes=loss_dict["gt_classes"],
                        gt_origin=loss_dict["gt_origin"],
                        gt_target_idx=dn_init.gt_target_idx,
                    )
                    loss_dict["total"] = loss_dict["total"] + dn_dict["total"]
                    for k, v in dn_dict.items():
                        if k == "total":
                            continue
                        loss_dict[f"dn_{k}"] = v
                    # Strip the non-scalar tensors / numpy arrays that
                    # return_matching=True added — the training-mode
                    # aggregator only handles 0-d tensors.
                    for k in ("q_idx", "k_idx", "per_level_gt_mask",
                              "gt_classes", "gt_origin",
                              "gt_truth_indices"):
                        loss_dict.pop(k, None)
                # Dense keypoint head loss (per-SP score MSE + offset vote),
                # supervised by the dataset's per-SP kpscores / kpoffsets.
                if (dense_logits_all is not None
                        and "kpscores" in data_dict):
                    d_logits = dense_logits_all[sp]
                    d_off = dense_off_all[sp] if dense_off_all is not None else None
                    d_tgt = data_dict["kpscores"][sp]
                    d_tgt_off = (data_dict["kpoffsets"][sp]
                                 if "kpoffsets" in data_dict else None)
                    d_loss, d_diag = self._dense_keypoint_loss(
                        d_logits, d_off, d_tgt, d_tgt_off)
                    loss_dict["total"] = loss_dict["total"] + d_loss
                    for k, v in d_diag.items():
                        loss_dict[f"kpdense_{k}"] = v
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
                        keypoint_refine_layers=refine_layers,
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
