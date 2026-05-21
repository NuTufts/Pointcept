"""PTv3DecoderStageLevel — LevelBuilder backed by one stage of PT-v3m2's
learned decoder.

Unlike VoxelBuilder (which mean-pools backbone-final per-SP features into
a user-specified voxel grid), this builder consumes features produced AT
the encoder's natural pyramid stride, after the PTv3 decoder's transformer
blocks have refined them at that scale. It's the "use PTv3's pixel-decoder
analog" path for Mask2Former-style multi-scale supervision — see
`docs/LArFormer.md` §15.

Requires LArFormer to be built with `capture_decoder_stages=True` and the
inner PT-v3m2 backbone to be built with `enc_mode=False` (and the Sonata-
v1m1 wrapper, if used, with `up_cast_level=0` so it doesn't try to upcast
the already-decoded output).

Per-event data is precomputed by `LArFormer._build_decoder_stages_per_event`
and lives under `event_dict["ptv3_dec_stages"][stage_key]` as a dict with
keys `{"tokens", "coords", "sp_to_level_id"}`.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import BUILDERS, LevelBuilder, LevelOutput


@BUILDERS.register_module()
class PTv3DecoderStageLevel(LevelBuilder):
    """Level whose tokens are PT-v3m2 decoder-stage outputs.

    Args:
        in_dim:      channel dim of the captured stage Point.feat. This is
                      `dec_channels[stage]` from the PT-v3m2 config — pass
                      it explicitly in `builder_cfg` since it differs from
                      the model's global `backbone_out_channels`.
        token_dim:   decoder/refiner token dim (matches LArFormer.token_dim).
        stage_key:   which decoder stage to read, e.g. "dec1" (stride 2),
                      "dec2" (stride 4), "dec3" (stride 8). Must match a
                      key emitted by `LArFormer._build_decoder_stages_per_event`.
        norm:        optional normalization on the projected tokens.
                      None (default) | "layer" | "batch".
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        stage_key: str,
        norm: Optional[str] = None,
    ):
        super().__init__(in_dim, token_dim)
        self.stage_key = str(stage_key)
        self.proj = nn.Linear(in_dim, token_dim)
        if norm is None:
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.LayerNorm(token_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(token_dim)
        else:
            raise ValueError(
                f"PTv3DecoderStageLevel norm must be None / 'layer' / "
                f"'batch'; got {norm!r}"
            )

    def forward(
        self,
        sp_feat: torch.Tensor,         # ignored — our tokens come from event_dict
        coord_norm: torch.Tensor,       # ignored
        event_dict: dict,
    ) -> LevelOutput:
        stages = event_dict.get("ptv3_dec_stages")
        if stages is None:
            raise RuntimeError(
                f"PTv3DecoderStageLevel({self.stage_key}): event_dict has "
                f"no 'ptv3_dec_stages' key. Build LArFormer with "
                f"capture_decoder_stages=True and configure the inner PT-"
                f"v3m2 backbone with enc_mode=False so its decoder runs."
            )
        if self.stage_key not in stages:
            raise KeyError(
                f"PTv3DecoderStageLevel({self.stage_key}): stage "
                f"{self.stage_key!r} not in captured stages "
                f"{sorted(stages.keys())!r}. Check that the backbone's "
                f"dec_depths covers this stride."
            )
        ev = stages[self.stage_key]
        tokens = ev["tokens"]                              # (M, in_dim)
        if tokens.shape[0] == 0:
            return LevelOutput(
                tokens=tokens.new_zeros(0, self.token_dim),
                coords=ev["coords"],
                sp_to_level_id=ev["sp_to_level_id"],
                name=self.name,
            )
        if tokens.shape[1] != self.in_dim:
            raise RuntimeError(
                f"PTv3DecoderStageLevel({self.stage_key}): expected "
                f"in_dim={self.in_dim} but stage tokens have width "
                f"{tokens.shape[1]}. Check dec_channels in the backbone "
                f"config matches the builder_cfg in_dim."
            )
        feat = self.norm(self.proj(tokens))               # (M, token_dim)
        return LevelOutput(
            tokens=feat,
            coords=ev["coords"],
            sp_to_level_id=ev["sp_to_level_id"],
            name=self.name,
        )
