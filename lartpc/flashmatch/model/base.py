"""
FlashModel: detector-agnostic flash prediction and scoring.

    PE_d = G · Σ_P n(P) · [ v_dir,d(P') · ε_dir,d + v_refl,d(P') · ε_refl,d ]

  P'      drift-corrected position: x' = x + drift_dir_x(TPC of P) · v_drift · t_flash
  n(P)    photons (or a photon proxy) per point, from the detector's charge model
  v_*     visibility model (photon library, semi-analytic, ...); v_refl may be absent
  ε_*     per-channel efficiencies (1 / 0 when the visibility already includes them)
  G       global light-yield scale (calibrated gamma x readout factor)

Detector subclasses (detectors/*.py) supply the pieces; this class owns the
arithmetic, so every detector predicts, drift-corrects and scores the same way.
See ../SBND_SEMIANALYTIC_PLAN.md for the design.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ScoreConfig:
    """How predicted and observed PE are compared.

    kind="neyman" (default, all detectors):
        χ²_i = (obs − pred)² / (obs + (f_sys·obs)² + eps), summed over unmasked channels.
    kind="normalized" (sbncode QLLMatch, QLLMode 0 + NormalizeFlash):
        both vectors divided by their own sums (after zeroing masked channels), then
        χ²_i = (O − H)² / (H + (chi_error·H)²) with O,H floored at `floor`, averaged over the
        channels used (optionally only those whose TPC matches `tpc`).
    """
    kind: str = "neyman"
    f_sys: float = 0.10
    eps: float = 1.0
    chi_error: float = 0.1
    floor: float = 1e-6


def normalized_chi2(pe_pred, pe_obs, mask=None, chi_error=0.1, floor=1e-6,
                    channel_tpc=None, tpc=None):
    """sbncode flashmatch::QLLMatch::QLL in kChi2 mode with NormalizeFlash (see ScoreConfig)."""
    H = np.asarray(pe_pred, dtype=np.float64).copy()
    O = np.asarray(pe_obs, dtype=np.float64).copy()
    use = np.ones(H.shape[0], dtype=bool)
    if mask is not None:
        m = np.asarray([int(c) for c in mask if 0 <= int(c) < H.shape[0]], dtype=np.int64)
        H[m] = 0.0
        O[m] = 0.0
        use[m] = False
    hs, os_ = H.sum(), O.sum()
    if hs != 0:
        H /= hs
    if os_ != 0:
        O /= os_
    if channel_tpc is not None and tpc is not None:
        use &= (np.asarray(channel_tpc) == int(tpc))
    O = np.where(O < floor, floor, O)
    H = np.where(H <= floor, floor, H)
    per = (O - H) ** 2 / (H + (chi_error * H) ** 2)
    n = int(use.sum())
    return float(per[use].sum() / n) if n else float("nan")


class FlashModel:
    """Base class. Subclasses set the attributes below in __init__ and override
    `visibility_torch` (and `photons` when the charge model is not the identity)."""

    detector = "base"
    #: number of optical channels in the predicted vectors
    n_channels: int = 0
    #: per-TPC sign of the drift direction along x, and how to find a point's TPC
    drift_dir_x: Sequence[float] = (-1.0,)
    #: active volumes [(min xyz, max xyz)], one per TPC (for TPC lookup and OOB)
    active_boxes: Sequence = ()
    v_drift_cm_per_us: float = 0.0
    #: channel -> TPC index (for the normalized score); None = all channels count
    channel_tpc: Optional[np.ndarray] = None

    def __init__(self, score: ScoreConfig = None, device=None):
        self.score_cfg = score or ScoreConfig()
        self._device = device
        self.eff_direct = None          # torch (D,) or None (= 1)
        self.eff_reflected = None       # torch (D,) or None (= no reflected term)

    # ----------------------------------------------------------- pieces
    def visibility_torch(self, pos):
        """(N,3) torch positions (drift-corrected) -> (direct (N,D), reflected (N,D) | None)."""
        raise NotImplementedError

    def photons(self, charge, points=None, **kw):
        """Charge -> photon proxy n(P). Default: identity (float32)."""
        return np.asarray(charge, dtype=np.float32)

    @property
    def device(self):
        raise NotImplementedError

    @property
    def vis_dtype(self):
        import torch
        return torch.float32

    # ----------------------------------------------------------- geometry
    def tpc_of(self, pos_cm):
        """(N,) TPC index of each point from its x position. Points are assigned by x only
        (y/z outliers still drift with their TPC); x outside every TPC -> nearest TPC in x."""
        p = np.asarray(pos_cm, dtype=np.float64).reshape(-1, 3)
        out = np.full(p.shape[0], -1, dtype=np.int64)
        if len(self.active_boxes) == 1:
            out[:] = 0                  # single drift volume: everything drifts the same way
            return out
        for t, (lo, hi) in enumerate(self.active_boxes):
            m = (out < 0) & (p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0])
            out[m] = t
        miss = out < 0
        if miss.any():                  # outside every x range: nearest TPC in x
            cx = np.array([0.5 * (lo[0] + hi[0]) for lo, hi in self.active_boxes])
            out[miss] = np.argmin(np.abs(p[miss, 0:1] - cx[None, :]), axis=1)
        return out

    def drift_shift_cm(self, pos_cm, flash_t0_us):
        """(N,) x shift to add: drift_dir_x(TPC) · v_drift · t_flash."""
        tpc = self.tpc_of(pos_cm)
        sign = np.asarray(self.drift_dir_x, dtype=np.float64)[np.clip(tpc, 0, None)]
        return sign * float(self.v_drift_cm_per_us) * float(flash_t0_us)

    def drift_correct(self, pos_cm, flash_t0_us):
        """numpy copy with x_true = x_app + drift_dir_x · v · t (float32)."""
        out = np.asarray(pos_cm, dtype=np.float32).copy()
        if out.size:
            out[:, 0] = out[:, 0] + self.drift_shift_cm(pos_cm, flash_t0_us).astype(np.float32)
        return out

    def oob_mask(self, pos_cm, margin_cm=0.0):
        """True for points outside every active volume (positions already drift-corrected)."""
        p = np.asarray(pos_cm, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(p.shape[0], dtype=bool)
        for lo, hi in self.active_boxes:
            lo = np.asarray(lo) + margin_cm
            hi = np.asarray(hi) - margin_cm
            inside |= np.all((p >= lo) & (p <= hi), axis=1)
        return ~inside

    def oob_fraction(self, pos_cm, flash_t0_us, margin_cm=0.0):
        if len(pos_cm) == 0:
            return 1.0
        return float(self.oob_mask(self.drift_correct(pos_cm, flash_t0_us), margin_cm).mean())

    # ----------------------------------------------------------- prediction
    def _drift_correct_torch(self, pos, pos_cm, flash_t0_us):
        import torch
        pos = pos.clone()
        shift = self.drift_shift_cm(pos_cm, flash_t0_us)
        uniform = shift.size == 0 or bool(np.all(shift == shift[0]))
        if uniform:
            # one drift volume (MicroBooNE): subtract a python float from the float32
            # column, the exact operation of the legacy flash_predict code
            s = float(shift[0]) if shift.size else 0.0
            pos[:, 0] = pos[:, 0] + s if s >= 0 else pos[:, 0] - (-s)
        else:
            pos[:, 0] = pos[:, 0] + torch.as_tensor(shift, dtype=pos.dtype, device=pos.device)
        return pos

    def predict(self, pos_cm, charge, cluster_id=None, n_clusters=1, flash_t0_us=0.0,
                light_yield=1.0, charge_is_photons=False, **photon_kw):
        """Predicted PE per cluster: (n_clusters, n_channels) float32.

        pos_cm:      (N,3) RAW (apparent) positions; drift correction is applied here.
        charge:      (N,) per-point charge in the detector's charge-model units.
        cluster_id:  (N,) int in [0, n_clusters) (None -> one cluster).
        light_yield: G, the global scale (e.g. calibrated gamma x readout factor).
        charge_is_photons: `charge` already holds photons per point (skip the charge model,
                     e.g. for MC true energy deposits x light yield).
        """
        import torch
        pos_cm = np.asarray(pos_cm)
        n_clusters = int(n_clusters)
        if pos_cm.shape[0] == 0:
            return np.zeros((n_clusters, self.n_channels), dtype=np.float32)
        q_np = (np.asarray(charge, dtype=np.float32) if charge_is_photons
                else self.photons(charge, pos_cm, **photon_kw))
        if q_np.shape[0] != pos_cm.shape[0]:
            raise ValueError(f"pos_cm and charge length mismatch: {pos_cm.shape[0]} vs {q_np.shape[0]}")
        dev, dt = self.device, self.vis_dtype
        pos = torch.from_numpy(pos_cm.astype(np.float32, copy=False)).to(dev)
        pos = self._drift_correct_torch(pos, pos_cm, flash_t0_us).to(dt)
        q = torch.from_numpy(np.asarray(q_np).astype(np.float32, copy=False)).to(dev)
        if cluster_id is None:
            cid = torch.zeros(pos.shape[0], dtype=torch.int64, device=dev)
        else:
            cid = torch.from_numpy(np.asarray(cluster_id).astype(np.int64, copy=False)).to(dev)

        vd, vr = self.visibility_torch(pos)
        vis = vd if self.eff_direct is None else vd * self.eff_direct.to(vd.dtype)
        if vr is not None and self.eff_reflected is not None:
            vis = vis + vr * self.eff_reflected.to(vr.dtype)
        contrib = q.to(vis.dtype).unsqueeze(-1) * vis
        out = torch.zeros((n_clusters, vis.shape[1]), dtype=vis.dtype, device=dev)
        out.index_add_(0, cid, contrib)
        return (out * float(light_yield)).detach().cpu().numpy().astype(np.float32)

    def predict_slice(self, pos_cm, charge, flash_t0_us=0.0, light_yield=1.0, **kw):
        """(n_channels,) prediction for one slice."""
        return self.predict(pos_cm, charge, None, 1, flash_t0_us, light_yield, **kw)[0]

    # ----------------------------------------------------------- masks + score
    def channel_mask(self, pe_obs, run=None, **kw):
        """Channels to exclude from the score for this event (detector-specific)."""
        return ()

    def chi2(self, pe_pred, pe_obs, mask=None, tpc=None, return_per_channel=False):
        """Score one predicted vector against the observed one.
        mask: channels to exclude. tpc: if given (and channel_tpc is set), only that
        TPC's channels are scored (e.g. SBND per-TPC flashes)."""
        cfg = self.score_cfg
        if cfg.kind == "neyman":
            from ..flash_chi2 import neyman_chi2
            if tpc is not None and self.channel_tpc is not None:
                other = np.nonzero(np.asarray(self.channel_tpc) != int(tpc))[0]
                mask = tuple(sorted(set(mask or ()) | set(int(c) for c in other)))
            return neyman_chi2(pe_pred, pe_obs, f_sys=cfg.f_sys, eps=cfg.eps,
                               dead_opdets=mask, return_per_pmt=return_per_channel)
        if cfg.kind == "normalized":
            if return_per_channel:
                raise NotImplementedError("per-channel terms only for the neyman score")
            return normalized_chi2(pe_pred, pe_obs, mask=mask, chi_error=cfg.chi_error,
                                   floor=cfg.floor, channel_tpc=self.channel_tpc, tpc=tpc)
        raise ValueError(f"unknown score kind {cfg.kind!r}")
