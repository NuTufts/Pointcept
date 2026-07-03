"""CascadedParticleSegmenter — Stage 1+2+3 cascade.

Wraps a `CascadedSlicer` (Stages 1 + 2) and a Stage-3 particle-segmenter
`LArFormer`. The slicer cascade ALWAYS runs in eval mode (returns
predictions, not its own loss). Per-SP nu-candidate masks are built from
the slicer's predictions with a **loosened** mask-prob threshold
(`mask_prob_threshold`, default 0.5), then the post-deghoster batch is
filtered down to those SPs and (optionally) recentered to the per-event
slice centroid before being handed to the particle segmenter.

See `docs/reference/LArFormer_particlesegment_stage.md` for the full design
(§1b cascade location, §3 input-set definition, §1f recentering).

Cascade composition:

    raw spacepoints + GT instances (particle-level, from LArFormerDataset
        with gt_source="particle")
        │
        ▼  self.cascaded_slicer  (always eval; freeze_cascaded_slicer
                                   controls torch.no_grad context)
    Stage-2 predictions per event (class_logits + mask_logits) +
    post-deghoster `filtered_batch`
        │
        ▼  build_nu_keep_mask(...): union of model-nu queries'
                                      `mask_prob > τ_loose` per SP
        ▼  filter_batch_for_particle_segmenter(...): filter + (optional)
                                      coord_norm recentering to per-event
                                      nu-SP centroid
    Stage-3 input batch (per-SP tensors restricted to the nu candidate
    set, gt_instances filtered to particles whose truth_indices survive
    the filter)
        │
        ▼  self.particle_segmenter  (the only training-mode forward)
    particle instance masks + per-query particle-class logits +
    per-particle origin (start point)

Empty-batch handling mirrors `CascadedSlicer`: if the slicer drops every
event OR no nu queries fire OR no SPs survive τ_loose, the wrapper
returns a graph-connected zero loss in training and an empty predictions
list in eval. Same warning semantics.

Weight-loading entry points (independent):
- `cascaded_slicer_weight`     — checkpoint of a trained CascadedSlicer
                                  (i.e. one that contains BOTH deghoster
                                  and slicer state under matching keys).
- `cascaded_slicer.deghoster_weight` / `cascaded_slicer.slicer_backbone_weight`
                                — alternatively load the deghoster + slicer
                                  pretrains independently via the inner
                                  CascadedSlicer's own knobs (preferred
                                  in v1 since we already have those
                                  artifacts).
- `particle_segmenter_backbone_weight`
                                — Sonata pretrain checkpoint for the
                                  Stage-3 LArFormer's backbone.
"""

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import warnings

from pointcept.models.builder import MODELS, build_model

from .cascade_particle_filter import (
    build_nu_keep_mask,
    build_forced_keep_mask,
    filter_batch_for_particle_segmenter,
    slicer_predictions_empty,
)
from .cascade_filter import drop_empty_events


@MODELS.register_module()
class CascadedParticleSegmenter(nn.Module):
    """Stage 1+2+3 cascade wrapping a CascadedSlicer + a Stage-3 LArFormer.

    Args:
        cascaded_slicer:           subconfig — typically `CascadedSlicer`.
                                    Always runs in eval mode regardless
                                    of the outer model's mode.
        particle_segmenter:        subconfig — typically `LArFormer` with
                                    K ≈ 32 queries, particle-class taxonomy,
                                    `enable_origin_head=True`, mask_denoising
                                    enabled.
        cascaded_slicer_weight:    optional path to a CascadedSlicer
                                    checkpoint. Loaded with strict=False.
        particle_segmenter_backbone_weight:
                                    optional Sonata pretrain for the
                                    particle segmenter's backbone only.
        freeze_cascaded_slicer:    if True (default), the slicer cascade
                                    forward runs under `torch.no_grad`
                                    and its params have requires_grad=False.
        nu_class_id:               which column of the slicer's class_argmax
                                    counts as "nu" (default 0; matches the
                                    Stage-2 model's nu_class_id).
        mask_prob_threshold:       τ_loose for nu mask membership (default
                                    0.5). Lower = more SPs survive into
                                    Stage 3 (recovers under-clustered SPs
                                    per §3 of the plan).
        spacepoint_level:          name of the SP-level entry in the slicer's
                                    `mask_logits` OrderedDict. Default
                                    "spacepoint".
        recenter_to_slice_centroid: subtract per-event nu-SP centroid from
                                    coord_norm before Stage 3 (§1f).
                                    Default True.
        report_keep_frac:          forward `cascaded_slicer`'s τ /
                                    keep_frac AND a Stage-3 nu_keep_frac
                                    scalar in the loss dict for monitoring.
    """

    def __init__(
        self,
        cascaded_slicer: dict,
        particle_segmenter: dict,
        cascaded_slicer_weight: Optional[str] = None,
        particle_segmenter_backbone_weight: Optional[str] = None,
        freeze_cascaded_slicer: bool = True,
        nu_class_id: int = 0,
        mask_prob_threshold: float = 0.5,
        spacepoint_level: str = "spacepoint",
        recenter_to_slice_centroid: bool = True,
        report_keep_frac: bool = True,
        expose_filtered_batch: bool = False,
    ):
        super().__init__()
        self.cascaded_slicer = build_model(cascaded_slicer)
        self.particle_segmenter = build_model(particle_segmenter)

        if cascaded_slicer_weight is not None:
            self._load_cascaded_slicer_weight(cascaded_slicer_weight)
        if particle_segmenter_backbone_weight is not None:
            self._load_particle_segmenter_backbone_weight(
                particle_segmenter_backbone_weight,
            )

        self.freeze_cascaded_slicer = bool(freeze_cascaded_slicer)
        self.nu_class_id = int(nu_class_id)
        self.mask_prob_threshold = float(mask_prob_threshold)
        self.spacepoint_level = str(spacepoint_level)
        self.recenter_to_slice_centroid = bool(recenter_to_slice_centroid)
        self.report_keep_frac = bool(report_keep_frac)
        # When set, the eval return also carries the recentered nu-slice batch
        # (`ps_batch`) so a downstream model (e.g. CascadedKeypoint) can run its
        # own backbone on the exact slice Stage 3 saw.
        self.expose_filtered_batch = bool(expose_filtered_batch)

        if self.freeze_cascaded_slicer:
            for p in self.cascaded_slicer.parameters():
                p.requires_grad_(False)

        # Pin the slicer cascade to eval from construction. train() override
        # below maintains the invariant across train/eval flips.
        self.cascaded_slicer.eval()

    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override: keep the slicer cascade permanently in eval mode.

        The slicer cascade's forward path is only useful as inference for
        Stage 3 — we want its `predictions` dict, not its training loss.
        Mirrors `CascadedSlicer.train()` which keeps the deghoster eval.
        """
        super().train(mode)
        self.cascaded_slicer.eval()
        return self

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_state_dict_into(target: nn.Module, weight_path: str,
                              log_name: str) -> None:
        """Same prefix-stripping convention as CascadedSlicer._load_state_dict_into."""
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
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
        # Shape-aware filter — same defensive behaviour as
        # CascadedSlicer._load_state_dict_into.
        target_shapes = {k: tuple(v.shape) for k, v
                         in target.state_dict().items()}
        filtered_sd = {}
        n_shape_mismatch = 0
        first_mismatches = []
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
        missing, unexpected = target.load_state_dict(filtered_sd, strict=False)
        print(f"[CascadedParticleSegmenter] Loaded {log_name} weight: "
              f"{weight_path}")
        print(f"  missing={len(missing)}  unexpected={len(unexpected)}  "
              f"shape_mismatch_dropped={n_shape_mismatch}")
        if missing[:5]:
            print(f"  first missing: {missing[:5]}")
        if unexpected[:5]:
            print(f"  first unexpected: {unexpected[:5]}")
        if first_mismatches:
            print(f"  first shape-mismatch: {first_mismatches}")

    def _load_cascaded_slicer_weight(self, weight_path: str) -> None:
        self._load_state_dict_into(
            self.cascaded_slicer, weight_path, "cascaded_slicer",
        )

    def _load_particle_segmenter_backbone_weight(self, weight_path: str) -> None:
        """Load a Sonata pretrain into the Stage-3 particle segmenter's
        backbone only — leaves tokenizer / decoder / heads at their
        random init."""
        self._load_state_dict_into(
            self.particle_segmenter.backbone, weight_path,
            "particle_segmenter backbone",
        )

    # ------------------------------------------------------------------
    # Empty / degenerate batch helpers (mirror CascadedSlicer's pattern)
    # ------------------------------------------------------------------

    def _empty_train_output(self, device, slicer_out) -> dict:
        zero = sum(
            (p * 0.0).sum()
            for p in self.particle_segmenter.parameters() if p.requires_grad
        )
        if not isinstance(zero, torch.Tensor):
            zero = torch.zeros((), device=device, requires_grad=True)
        out = {"loss": zero}
        if self.report_keep_frac:
            out["nu_keep_frac"] = torch.tensor(0.0, device=device)
            for k in ("deghost_tau", "deghost_keep_frac"):
                if k in slicer_out:
                    v = slicer_out[k]
                    out[k] = (v if isinstance(v, torch.Tensor)
                              else torch.tensor(float(v), device=device))
        return out

    def _empty_eval_output(self, slicer_out) -> dict:
        out = {
            "predictions": [],
            "nu_keep_frac": 0.0,
            "slicer_predictions": slicer_out.get("predictions", []),
            "deghost_tau": slicer_out.get("deghost_tau"),
            "deghost_keep_frac": slicer_out.get("deghost_keep_frac"),
            "deghost_p_real": slicer_out.get("deghost_p_real"),
        }
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict: dict) -> dict:
        device = data_dict["coord_norm"].device

        # ---- Stage 1+2: deghoster + slicer (always eval) -------------
        # Strip gt_instances_per_event before handing to the inner
        # slicer. Reason: this dict carries PARTICLE-level GT (Stage-3's
        # 7-class taxonomy + per-particle origin) which the slicer's
        # LArFormer.forward eval-with-GT branch can't consume — its loss
        # function would try to compute CE against a 3-class slot with a
        # target in {0..6} and CUDA-assert. We only need the slicer's
        # PREDICTIONS here, not its eval loss, so dropping the GT skips
        # that branch cleanly.
        # Eval-only: a caller may inject a pre-computed slicer output so several
        # Stage-3 passes (e.g. per-slice) share ONE slicer forward -- the slicer's
        # query->slice assignment is NOT stable across separate forwards, so
        # re-running it per pass would make each pass see a different slice.
        forced_sl = getattr(self, "_force_slicer_out", None)
        if (not self.training) and forced_sl is not None:
            slicer_out = forced_sl
        else:
            slicer_input = {k: v for k, v in data_dict.items()
                            if k not in ("gt_instances_per_event",
                                         "n_gt_instances")}
            ctx = (torch.no_grad() if self.freeze_cascaded_slicer
                   else nullcontext())
            self.cascaded_slicer.eval()
            with ctx:
                slicer_out = self.cascaded_slicer(slicer_input)

        slicer_predictions = slicer_out.get("predictions", [])
        filtered_batch = slicer_out.get("filtered_batch", None)
        if filtered_batch is None or slicer_predictions_empty(slicer_predictions):
            warnings.warn(
                "CascadedParticleSegmenter: slicer cascade produced no "
                "usable predictions (empty batch or no decoder output). "
                "Stage 3 is being skipped for this batch."
            )
            return (self._empty_train_output(device, slicer_out)
                    if self.training
                    else self._empty_eval_output(slicer_out))

        # ---- Stage 2 → 3 boundary -----------------------------------
        # Build the per-SP nu-keep mask from the slicer's predictions -- UNLESS a
        # caller injected an explicit keep mask (eval-only, e.g. an inference tool
        # running Stage 3 on a chosen slice instead of the nu union). The forced
        # mask must be aligned with `filtered_batch["offset"]`; deterministic eval
        # (fixed deghost tau) guarantees the same filtered batch on a re-run.
        forced = getattr(self, "_force_slice", None)
        if (not self.training) and forced is not None:
            keep = build_forced_keep_mask(
                slicer_predictions=slicer_predictions,
                n_sp_per_event=filtered_batch["n_spacepoints"],
                spec=forced, nu_class_id=self.nu_class_id,
                mask_prob_threshold=self.mask_prob_threshold,
                spacepoint_level=self.spacepoint_level, device=device)
        else:
            keep = build_nu_keep_mask(
                slicer_predictions=slicer_predictions,
                n_sp_per_event=filtered_batch["n_spacepoints"],
                nu_class_id=self.nu_class_id,
                mask_prob_threshold=self.mask_prob_threshold,
                spacepoint_level=self.spacepoint_level,
                device=device,
            )
        keep_frac = (float(keep.float().mean().item())
                     if keep.numel() > 0 else 0.0)

        # If no SP survives nu-mask, return graceful empty output.
        if int(keep.sum().item()) == 0:
            warnings.warn(
                "CascadedParticleSegmenter: zero SPs passed τ_loose "
                f"({self.mask_prob_threshold}) over the slicer's nu "
                "queries. Returning empty Stage-3 output."
            )
            empty = (self._empty_train_output(device, slicer_out)
                     if self.training
                     else self._empty_eval_output(slicer_out))
            empty["nu_keep_frac"] = (
                torch.tensor(keep_frac, device=device)
                if self.training else keep_frac
            )
            return empty

        ps_batch = filter_batch_for_particle_segmenter(
            filtered_batch=filtered_batch,
            keep_mask=keep,
            recenter=self.recenter_to_slice_centroid,
            drop_empty_instances=True,
        )

        # After filtering, some events may have 0 SPs surviving — drop
        # them so the Stage-3 model sees a clean batch.
        if int((ps_batch["n_spacepoints"] == 0).any().item()):
            ps_batch = drop_empty_events(ps_batch)
        if ps_batch["n_spacepoints"].numel() == 0 \
                or int(ps_batch["n_spacepoints"].sum().item()) == 0:
            warnings.warn(
                "CascadedParticleSegmenter: every event lost all SPs "
                "after nu-filter + drop_empty. Skipping Stage 3."
            )
            empty = (self._empty_train_output(device, slicer_out)
                     if self.training
                     else self._empty_eval_output(slicer_out))
            empty["nu_keep_frac"] = (
                torch.tensor(keep_frac, device=device)
                if self.training else keep_frac
            )
            return empty

        # ---- Stage 3 ------------------------------------------------
        ps_out = self.particle_segmenter(ps_batch)

        # ---- Merge / decorate outputs --------------------------------
        if self.training:
            out: dict = {"loss": ps_out["loss"]}
            for k, v in ps_out.items():
                if k == "loss":
                    continue
                out[f"ps_{k}"] = v
            if self.report_keep_frac:
                out["nu_keep_frac"] = torch.tensor(keep_frac, device=device)
                for k in ("deghost_tau", "deghost_keep_frac"):
                    if k in slicer_out:
                        v = slicer_out[k]
                        out[k] = (v if isinstance(v, torch.Tensor)
                                  else torch.tensor(float(v), device=device))
            return out

        # Eval — pass through Stage 3's predictions, decorate with
        # Stage-1+2 telemetry and the nu-filter survivor info so a
        # full-cascade evaluator can correlate per-particle metrics with
        # which nu-slice candidate Stage 2 chose.
        out = {
            "predictions": ps_out["predictions"],
            "nu_keep_frac": keep_frac,
            "slicer_predictions": slicer_predictions,
            "deghost_tau": slicer_out.get("deghost_tau"),
            "deghost_keep_frac": slicer_out.get("deghost_keep_frac"),
            "deghost_p_real": slicer_out.get("deghost_p_real"),
            "filtered_batch_n_spacepoints":
                ps_batch.get("n_spacepoints"),
            "filtered_gt_instances_per_event":
                ps_batch.get("gt_instances_per_event", []),
        }
        if self.expose_filtered_batch:
            # The recentered nu-slice batch Stage 3 ran on (coord_norm/feat/
            # offset/grid_coord/gt_instances/...) — for a downstream keypoint
            # model to run its own backbone on the same slice.
            out["ps_batch"] = ps_batch
        return out
