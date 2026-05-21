"""
CascadedSlicer — Stage-2 of the LArFormer cascade (see
`Pointcept/docs/LArFormer.md` §6).

Composition (not inheritance) of two `LArFormer` instances:

    raw spacepoints
        │
        ▼  self.deghoster  (frozen by default; LArFormer with num_queries=0
                            + per-token cls head on `hasmatch`)
    per-SP P(real) score
        │  sample τ; build keep_mask = P(real) > τ
        ▼  filter_batch_by_keep_mask
    deghosted spacepoints (+ remapped truth_indices)
        │
        ▼  self.slicer     (LArFormer with Mask2Former-style decoder +
                            multi-voxel + spacepoint scale_pattern)
    slice instance masks + per-query class logits

The deghoster is always called in eval mode (we want its per-SP predictions,
not its own training loss). Wrapping it in `torch.no_grad()` is controlled
by `freeze_deghoster` (default True). For v0 the threshold filter is a
hard step function, so end-to-end gradients DO NOT flow through to the
deghoster even if `freeze_deghoster=False`; the flag is exposed for future
work that adds a soft/differentiable selection.

Threshold sampling (per batch, your (3)): in train, τ ~ U(min, max); in
eval, τ = val. If after the initial draw any event has zero survivors,
re-sample τ from the lower half once; if still empty, drop those events
from the batch.
"""

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model

from .cascade_filter import drop_empty_events, filter_batch_by_keep_mask


@MODELS.register_module()
class CascadedSlicer(nn.Module):
    """Frozen-deghoster + slicer cascade.

    Args:
        deghoster:                  full LArFormer subconfig (built via the
                                     MODELS registry).
        deghoster_weight:           optional path to a checkpoint; loaded
                                     into `self.deghoster` at init via the
                                     same key-remap rule as SonataCheckpoint-
                                     Loader (state_dict keys are matched
                                     directly, with strict=False).
        slicer:                     full LArFormer subconfig (the Stage-2
                                     model).
        slicer_backbone_weight:     optional path to a pretrained Sonata
                                     backbone checkpoint; loaded into
                                     `self.slicer.backbone` only. Use this
                                     instead of the top-level `weight =`
                                     when the cascaded model has multiple
                                     backbones (SonataCheckpointLoader
                                     can't scope by submodule). The slicer's
                                     decoder / tokenizer / heads are not
                                     touched.
        deghost_threshold_min:      lower bound of τ ~ U(min, max) at train.
        deghost_threshold_max:      upper bound at train.
        deghost_threshold_val:      fixed τ at val/test.
        freeze_deghoster:           if True (default), deghoster params have
                                     requires_grad=False and forward runs
                                     under torch.no_grad. The deghoster is
                                     always run in .eval() mode regardless.
        deghoster_class_index_real: index of the "real" class in the deghoster's
                                     per-token cls head (default 1; matches
                                     `larformer-deghost-v0.py` where
                                     hasmatch={0=ghost, 1=real}).
        report_keep_frac:           if True, the training loss dict includes
                                     `keep_frac` and `tau` scalars for
                                     monitoring. Cheap; useful.
    """

    def __init__(
        self,
        deghoster: dict,
        slicer: dict,
        deghoster_weight: Optional[str] = None,
        slicer_backbone_weight: Optional[str] = None,
        deghost_threshold_min: float = 0.3,
        deghost_threshold_max: float = 0.7,
        deghost_threshold_val: float = 0.5,
        freeze_deghoster: bool = True,
        deghoster_class_index_real: int = 1,
        report_keep_frac: bool = True,
    ):
        super().__init__()
        if not (0.0 <= deghost_threshold_min <= deghost_threshold_max <= 1.0):
            raise ValueError(
                f"need 0 <= min ({deghost_threshold_min}) <= "
                f"max ({deghost_threshold_max}) <= 1"
            )
        if not (0.0 <= deghost_threshold_val <= 1.0):
            raise ValueError(
                f"deghost_threshold_val must be in [0, 1], got "
                f"{deghost_threshold_val}"
            )

        self.deghoster = build_model(deghoster)
        self.slicer = build_model(slicer)

        if deghoster_weight is not None:
            self._load_deghoster_weight(deghoster_weight)
        if slicer_backbone_weight is not None:
            self._load_slicer_backbone_weight(slicer_backbone_weight)

        self.deghost_threshold_min = float(deghost_threshold_min)
        self.deghost_threshold_max = float(deghost_threshold_max)
        self.deghost_threshold_val = float(deghost_threshold_val)
        self.freeze_deghoster = bool(freeze_deghoster)
        self.deghoster_class_index_real = int(deghoster_class_index_real)
        self.report_keep_frac = bool(report_keep_frac)

        if self.freeze_deghoster:
            for p in self.deghoster.parameters():
                p.requires_grad_(False)

        # Pin deghoster to eval mode from construction onward. train()
        # override below maintains this invariant across train/eval flips.
        self.deghoster.eval()

    # ------------------------------------------------------------------
    # State / mode management
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override: keep the deghoster permanently in .eval() mode.

        The deghoster's forward path is only useful as inference (we want
        its `predictions` dict, not its training-loss dict). For our
        frozen Sonata + tiny cls MLP, eval vs train mode is functionally a
        no-op (no dropout, no BN), but keeping eval is the right semantic
        contract and avoids surprises if the deghoster grows components
        later that DO behave differently in train/eval.
        """
        super().train(mode)
        self.deghoster.eval()
        return self

    @staticmethod
    def _load_state_dict_into(
        target_module: nn.Module,
        weight_path: str,
        log_name: str,
    ) -> None:
        """Load `weight_path`'s state_dict into `target_module`, with the
        standard DDP / SonataCheckpointLoader prefix-stripping rules.

        The two prefix forms we handle:
          - DDP "module." prefix    → always stripped (per-key)
          - Uniform "backbone." prefix → stripped only if EVERY remaining
            key starts with it (so per-key fields like `deghost_head.*`
            don't trigger a wrong strip)
        Other prefixes are left alone; if your checkpoint has a different
        prefix convention, pre-massage it with a small helper script or
        use a scoped CheckpointLoader hook instead.
        """
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
        missing, unexpected = target_module.load_state_dict(sd, strict=False)
        print(f"[CascadedSlicer] Loaded {log_name} weight: {weight_path}")
        print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
        if missing[:5]:
            print(f"  first missing: {missing[:5]}")
        if unexpected[:5]:
            print(f"  first unexpected: {unexpected[:5]}")

    def _load_deghoster_weight(self, weight_path: str) -> None:
        self._load_state_dict_into(self.deghoster, weight_path, "deghoster")

    def _load_slicer_backbone_weight(self, weight_path: str) -> None:
        """Load a vanilla Sonata pretrain (or LoRA-folded backbone) into the
        slicer's backbone only.

        The Sonata pretrain checkpoint has keys like `{teacher,student}.*`,
        which match `self.slicer.backbone` (a Sonata-v1m1) directly with
        `strict=False`. The Mask2Former-side modules of `self.slicer`
        (tokenizer, decoder, cls_heads, loss_fn) are NOT touched.
        """
        self._load_state_dict_into(
            self.slicer.backbone, weight_path, "slicer backbone",
        )

    # ------------------------------------------------------------------
    # Threshold sampling
    # ------------------------------------------------------------------

    def _sample_threshold(self) -> float:
        if self.training:
            return float(np.random.uniform(
                self.deghost_threshold_min, self.deghost_threshold_max,
            ))
        return self.deghost_threshold_val

    def _sample_threshold_lower_half(self) -> float:
        """Used as the retry on a τ that emptied at least one event."""
        mid = 0.5 * (self.deghost_threshold_min + self.deghost_threshold_max)
        return float(np.random.uniform(self.deghost_threshold_min, mid))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _run_deghoster_p_real(self, data_dict: dict) -> torch.Tensor:
        """Run the deghoster in eval mode; return flat per-SP P(real).

        Adapts the two deghoster output conventions we support:

            LArFormer-style (LArFormer with per-token cls head):
                out = {"predictions": [
                    {"per_level_cls": {"spacepoint": (n_sp_ev, 2)}}, ...
                ]}
              Per-event predictions; concatenated to a flat (N_total,) tensor.

            SonataLoRA-style (SonataLoRADeghostSegmentor, LoRA-finetuned
            Sonata + binary deghost head):
                out = {"seg_logits": (N_total, 2)}
              Already flat; just softmax + pick the real-class column.

        `deghoster_class_index_real` differs between the two:
          - LArFormer-flavored deghoster (hasmatch=1 = real): class 1
          - SonataLoRA deghoster (HasmatchAsGhost transform: real -> 0):
            class 0
        Set it in the model config to match whichever deghoster you load.
        """
        ctx = torch.no_grad() if self.freeze_deghoster else nullcontext()
        # Force eval-mode forward (which returns predictions, not loss).
        # We've already overridden train() to keep deghoster in eval, but
        # be defensive here for callers that may manipulate state.
        self.deghoster.eval()
        with ctx:
            out = self.deghoster(data_dict)

        if "seg_logits" in out:
            # SonataLoRA-style: flat (N_total, C) logits, no per-event slicing.
            return out["seg_logits"].softmax(dim=-1)[
                :, self.deghoster_class_index_real
            ]

        if "predictions" in out:
            # LArFormer-style: per-event predictions; concat to flat.
            chunks = []
            for ev_pred in out["predictions"]:
                logits = ev_pred["per_level_cls"]["spacepoint"]  # (n_sp_ev, C)
                p_real = logits.softmax(dim=-1)[
                    :, self.deghoster_class_index_real
                ]
                chunks.append(p_real)
            return torch.cat(chunks, dim=0)

        raise RuntimeError(
            f"Deghoster output dict has neither 'seg_logits' nor "
            f"'predictions' — got keys: {list(out.keys())}. Either the "
            f"deghoster ran in training mode (returning {{'loss': ...}}, "
            f"in which case fix CascadedSlicer.train() handling), or it's "
            f"a model class CascadedSlicer doesn't recognize. Add another "
            f"adapter branch in _run_deghoster_p_real."
        )

    def forward(self, data_dict: dict) -> dict:
        device = data_dict["coord_norm"].device

        # ---- Stage 1: deghoster ---------------------------------------
        p_real = self._run_deghoster_p_real(data_dict)

        # ---- Sample τ + build keep mask -------------------------------
        tau = self._sample_threshold()
        keep = p_real > tau
        filtered = filter_batch_by_keep_mask(data_dict, keep)

        # Empty-event guard: retry once at lower τ, then drop.
        if bool((filtered["n_spacepoints"] == 0).any()):
            tau = self._sample_threshold_lower_half()
            keep = p_real > tau
            filtered = filter_batch_by_keep_mask(data_dict, keep)
            if bool((filtered["n_spacepoints"] == 0).any()):
                filtered = drop_empty_events(filtered)

        keep_frac = float(keep.float().mean().item())

        # If after pruning the batch is empty, emit a graph-connected zero
        # loss so the trainer step is a defined no-op. Extraordinarily rare
        # given the retry; warn loudly so the user notices a misconfigured τ.
        if filtered["n_spacepoints"].numel() == 0 \
                or int(filtered["n_spacepoints"].sum().item()) == 0:
            import warnings
            warnings.warn(
                f"CascadedSlicer: every batch event dropped after deghoster "
                f"filter at tau={tau:.3f} (retry also empty). Threshold "
                f"range is probably too high."
            )
            if self.training:
                zero = sum(
                    (p * 0.0).sum()
                    for p in self.slicer.parameters() if p.requires_grad
                )
                if not isinstance(zero, torch.Tensor):
                    zero = torch.zeros((), device=device, requires_grad=True)
                return {
                    "loss": zero,
                    "deghost_tau": torch.tensor(tau, device=device),
                    "deghost_keep_frac": torch.tensor(keep_frac, device=device),
                }
            return {"predictions": [], "deghost_tau": tau,
                    "deghost_keep_frac": keep_frac}

        # ---- Stage 2: slicer ------------------------------------------
        slicer_out = self.slicer(filtered)

        # Re-wrap so the trainer sees a flat 0-d scalar dict; prefix every
        # slicer loss component with `slicer_` (deghoster contributes none
        # in v0). Track the deghost τ + keep fraction so wandb/log shows
        # whether the augmentation range is reasonable.
        if self.training:
            out: dict = {"loss": slicer_out["loss"]}
            for k, v in slicer_out.items():
                if k == "loss":
                    continue
                out[f"slicer_{k}"] = v
            if self.report_keep_frac:
                out["deghost_tau"] = torch.tensor(tau, device=device)
                out["deghost_keep_frac"] = torch.tensor(keep_frac, device=device)
            return out

        # Eval: return slicer predictions plus per-event deghost P(real).
        # Each entry of `predictions` now also carries an `eval_loss` field
        # when gt_instances were present (the slicer's LArFormer.forward
        # eval-with-GT branch adds it). The evaluator reads matching info
        # from there. We also pass through the post-filter per-event
        # identity lists so the evaluator can correlate predictions with
        # input events (drop_empty_events may have shrunk the batch).
        return {
            "predictions": slicer_out["predictions"],
            "deghost_tau": tau,
            "deghost_keep_frac": keep_frac,
            "deghost_p_real": p_real.detach(),
            "filtered_names": filtered.get("names", []),
            "filtered_runs": filtered.get("runs", []),
            "filtered_subruns": filtered.get("subruns", []),
            "filtered_events": filtered.get("events", []),
            "filtered_n_spacepoints": filtered["n_spacepoints"],
            # Post-filter gt_instances (mirrors gt_instances_per_event but
            # with empty / dropped instances pruned — what the slicer's
            # matcher actually saw). Downstream analyzers can index this
            # list with `eval_loss["k_idx"]` to get matched-GT metadata.
            "filtered_gt_instances_per_event":
                filtered.get("gt_instances_per_event", []),
        }
