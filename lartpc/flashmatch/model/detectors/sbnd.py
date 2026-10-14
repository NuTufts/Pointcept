"""
SBND flash model.

    PE_ch = G · Σ_P n(P) · [ v_dir,ch(P') · VUVEff_ch + v_refl,ch(P') · VISEff_ch ]

  v_*   inside the active volume: the semi-analytic model (port of larsim
        SemiAnalyticalModel, as SBND simulates in-AV light); outside: the SBND
        "OUT" photon library (as SBND simulates out-of-AV light), loaded lazily.
  Eff   per-channel VUV / visible efficiencies of SBND's flash matcher
        (opt0finder_sbnd.fcl, PhotonLibHypothesis), from detinfo.
  n(P)  electrons -> photons: q / CalAreaConstants[plane] · exp(t_drift / τ) · k_γ
        (SBNDOpT0Finder constant-conversion path; k_γ = ChargeToNPhotonsTrack = 1 by default).
  G     global light-yield scale (default 1; to be calibrated, plan phase P5).

Channels: all 312 optical channels are always in the output vectors (index =
optical channel = opdet, identity map in SBND); only the selected detector types
are computed, the rest are 0 and masked from the score. Default: PMTs only.

Known SBND v10_14_02 inconsistency: 6 VUV X-ARAPUCAs (channels 42, 44, 46, 49, 51,
53) sit, in the geometry, on the opposite side of the cathode from their channel
parity / PDS-map `tpc`. Visibilities follow the geometry (as the simulation does);
`channel_tpc` here also follows the geometry.
"""
import numpy as np
import torch

from .. import base
from ..visibility.semianalytic import SemiAnalyticVisibility

PMT_TYPES = ("pmt_coated", "pmt_uncoated")
ALL_TYPES = ("pmt_coated", "pmt_uncoated", "xarapuca_vuv", "xarapuca_vis")


class SBNDFlashModel(base.FlashModel):
    detector = "sbnd"

    def __init__(self, detinfo=None, detectors=PMT_TYPES, outside_library=True,
                 photons_per_electron=None, charge_plane=2, lifetime_correction=True,
                 light_yield=1.0, score=None, device=None):
        super().__init__(score=score, device=device)
        if detinfo is None:
            from lartpc.detinfo import load
            detinfo = load("sbnd")
        self.di = di = detinfo
        g = di.geometry
        om = di.optical_model
        fm = om["flashmatch"]["params"]
        hyp = fm["FlashMatchConfig"]["PhotonLibHypothesis"]
        self.qll_cfg = dict(fm["FlashMatchConfig"]["QLLMatch"])

        self.n_channels = g.NOpChannels()
        self.types = np.asarray(di.conventions.opdet_types(), dtype=object)
        if detectors == "all":
            detectors = ALL_TYPES
        unknown = set(detectors) - set(ALL_TYPES)
        if unknown:
            raise ValueError(f"unknown detector types {unknown}; choose from {ALL_TYPES} or 'all'")
        self.detector_types = tuple(detectors)
        self.selected = np.nonzero(np.isin(self.types, list(detectors)))[0]
        self.is_pmt = np.isin(self.types, list(PMT_TYPES))
        x = g.OpDetPositions()[:, 0]
        self.channel_tpc = (x > 0).astype(np.int64)            # geometry side (see module doc)

        # drift volumes
        tpcs = list(g.IterateTPCs())
        self.active_boxes = tuple((t.active_bbox.min.copy(), t.active_bbox.max.copy()) for t in tpcs)
        self.drift_dir_x = tuple(float(np.sign(t.drift_dir[0])) for t in tpcs)
        self.v_drift_cm_per_us = float(di.detprop.DriftVelocity())
        self.tpc_width = np.array([t.bbox.size[0] for t in tpcs])       # 2·HalfWidth per TPC
        self.lifetime_us = float(di.detprop.ElectronLifetime())

        # light model
        self._dev = torch.device(device if device is not None else
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sam = SemiAnalyticVisibility.from_detinfo(di, opdets=self.selected, device=self._dev)
        av = np.asarray(om["semianalytic"]["active_volumes"])[0]
        self.sam_av = (av[:3], av[3:])
        self.outside_library = bool(outside_library)
        self._lib = None
        self.eff_direct = torch.as_tensor(np.asarray(hyp["VUVEfficiency"], np.float64), device=self._dev)
        self.eff_reflected = torch.as_tensor(np.asarray(hyp["VISEfficiency"], np.float64), device=self._dev)
        self.global_qe = float(hyp.get("GlobalQE", 1.0))
        self.global_qe_refl = float(hyp.get("GlobalQERefl", 1.0))
        self.eff_direct = self.eff_direct * self.global_qe
        self.eff_reflected = self.eff_reflected * self.global_qe_refl

        # charge model
        self.cal_area = np.asarray(fm["CalAreaConstants"], np.float64)
        self.charge_plane = int(charge_plane)
        self.photons_per_electron = float(fm["ChargeToNPhotonsTrack"]
                                          if photons_per_electron is None else photons_per_electron)
        self.lifetime_correction = bool(lifetime_correction)
        self.default_light_yield = float(light_yield)

        # masks
        self.skip_channels = tuple(int(c) for c in np.asarray(fm["OpChannelsToSkip"]).astype(int))

    # ----------------------------------------------------------- torch plumbing
    @property
    def device(self):
        return self._dev

    @property
    def vis_dtype(self):
        return torch.float64

    @property
    def photonlib(self):
        if self._lib is None:
            from lartpc.detinfo.photonlib import PhotonLibrary
            import os
            name = self.di.photon_visibility_config.get("converted_h5")
            self._lib = PhotonLibrary(os.path.join(self.di.cache_dir, name), load_reflected=True)
        return self._lib

    def inside_av(self, pos):
        lo = torch.as_tensor(self.sam_av[0], dtype=pos.dtype, device=pos.device)
        hi = torch.as_tensor(self.sam_av[1], dtype=pos.dtype, device=pos.device)
        return torch.all((pos >= lo) & (pos <= hi), dim=1)

    def visibility_torch(self, pos):
        """(N,3) drift-corrected positions -> (direct, reflected), each (N, 312) float64."""
        N, C = pos.shape[0], self.n_channels
        vd = torch.zeros(N, C, dtype=torch.float64, device=pos.device)
        vr = torch.zeros(N, C, dtype=torch.float64, device=pos.device)
        sel = torch.as_tensor(self.selected, device=pos.device)
        inside = self.inside_av(pos)
        ii = torch.nonzero(inside).flatten()
        if ii.numel():
            d, r = self.sam.visibility(pos[ii])
            # larsim returns NaN reflected light for a point exactly on the cathode (0/0)
            r = torch.nan_to_num(r, nan=0.0)
            vd[ii[:, None], sel[None, :]] = d
            vr[ii[:, None], sel[None, :]] = r
        oo = torch.nonzero(~inside).flatten()
        if oo.numel() and self.outside_library:
            p = pos[oo].detach().cpu().numpy()
            lib = self.photonlib
            ld = torch.as_tensor(lib.GetVisibility(p).astype(np.float64), device=pos.device)
            lr = torch.as_tensor(lib.GetVisibility(p, reflected=True).astype(np.float64), device=pos.device)
            vd[oo[:, None], sel[None, :]] = ld[:, sel]
            vr[oo[:, None], sel[None, :]] = lr[:, sel]
        return vd, vr

    # ----------------------------------------------------------- charge model
    def photons(self, charge, points=None, plane=None, photons_per_electron=None, **kw):
        """Hit charge on `plane` (ADC area) -> photons:
        q / CalArea[plane] · exp(t_drift/τ) · k_γ, t_drift from the point's distance to its anode.
        photons_per_electron may be a scalar or a per-point array (e.g. track 1.0 / shower 1.25)."""
        q = np.asarray(charge, dtype=np.float64)
        plane = self.charge_plane if plane is None else int(plane)
        e = q / self.cal_area[plane]
        if self.lifetime_correction and points is not None and self.lifetime_us > 0:
            p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            tpc = self.tpc_of(p)
            dist = np.clip(self.tpc_width[tpc] - np.abs(p[:, 0]), 0.0, None)
            e = e * np.exp(dist / self.v_drift_cm_per_us / self.lifetime_us)
        k = self.photons_per_electron if photons_per_electron is None else photons_per_electron
        return (e * np.asarray(k, dtype=np.float64)).astype(np.float32)

    def predict(self, pos_cm, charge, cluster_id=None, n_clusters=1, flash_t0_us=0.0,
                light_yield=None, charge_is_photons=False, **photon_kw):
        G = self.default_light_yield if light_yield is None else light_yield
        return super().predict(pos_cm, charge, cluster_id, n_clusters, flash_t0_us, G,
                               charge_is_photons=charge_is_photons, **photon_kw)

    # ----------------------------------------------------------- masks + score
    def channel_mask(self, pe_obs=None, run=None, mask_saturated=True, max_saturated=4,
                     include_unselected=True):
        """skip list (opt0finder_sbnd OpChannelsToSkip) ∪ unselected detector types
        ∪ saturation holes among the selected PMTs (neighbour rule of saturation.py)."""
        m = set(self.skip_channels)
        if include_unselected:
            m |= set(int(c) for c in np.setdiff1d(np.arange(self.n_channels), self.selected))
        if mask_saturated and pe_obs is not None:
            from ... import saturation
            pm = np.array([c for c in self.selected if self.is_pmt[c] and c not in m])
            if pm.size >= 4:
                pos = self.di.geometry.OpDetPositions()[pm]
                holes = saturation.find_saturated(np.asarray(pe_obs)[pm], dead=(),
                                                  max_masked=int(max_saturated), positions=pos)
                m |= set(int(pm[i]) for i in holes)
        return tuple(sorted(m))

    def chi2(self, pe_pred, pe_obs, mask=None, tpc=None, return_per_channel=False):
        """Neyman (default) or, with score kind 'normalized', sbncode QLLMatch incl. its
        saturated-PMT and non-linearity fixes on the hypothesis."""
        if self.score_cfg.kind != "normalized":
            return super().chi2(pe_pred, pe_obs, mask, tpc, return_per_channel)
        H = np.asarray(pe_pred, dtype=np.float64).copy()
        O = np.asarray(pe_obs, dtype=np.float64)
        masked = np.zeros(H.shape[0], bool)
        if mask is not None:
            masked[[int(c) for c in mask]] = True
        sat_thr = float(self.qll_cfg.get("SaturatedThreshold", 0))
        if sat_thr > 0:
            pmt = self.is_pmt & ~masked
            H[pmt & (H >= sat_thr) & (O == 0)] = 0.0
            nl = pmt & (H > float(self.qll_cfg.get("NonLinearThreshold", np.inf))) & (O != 0)
            H[nl] = H[nl] * float(self.qll_cfg["NonLinearSlope"]) + float(self.qll_cfg["NonLinearOffset"])
        return base.normalized_chi2(H, O, mask=mask, chi_error=self.score_cfg.chi_error,
                                    floor=self.score_cfg.floor, channel_tpc=self.channel_tpc, tpc=tpc)
