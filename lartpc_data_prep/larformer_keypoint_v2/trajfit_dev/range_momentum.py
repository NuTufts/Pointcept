"""Range-based track 4-momentum (spec §3). Uses the ROOT-free Bethe-Bloch
range->KE tables (data/range2ke_lar.npz from make_range2ke_npz.py).
"""
import os

import numpy as np

MASS = {"e": 0.511, "gamma": 0.0, "mu": 105.6584, "pi": 139.5704, "p": 938.2721}
# predicted class -> (mass, range-table particle)
_TABLE = {"mu": ("mu", "muon"), "pi": ("pi", "pion"), "p": ("p", "proton")}


class RangeMomentum:
    def __init__(self, npz_path=None):
        if npz_path is None:
            npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data", "range2ke_lar.npz")
        self.t = {k: v for k, v in np.load(npz_path).items()}

    def ke(self, length_cm, particle):
        """KE [MeV] at track length [cm] for particle in {muon,pion,proton}."""
        return float(np.interp(length_cm, self.t[f"{particle}_range"],
                               self.t[f"{particle}_ke"]))

    def fourmom(self, length_cm, pred_class, direction):
        """Range 4-momentum for a track. Returns dict; the alt-hypothesis KEs are
        kept in `extra` for a PID cross-check (mu vs p, like LANTERN)."""
        d = np.asarray(direction, np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        m, table = _TABLE.get(pred_class, ("mu", "muon"))
        mass = MASS[m]
        ke = self.ke(length_cm, table)
        E = mass + ke
        p = float(np.sqrt(max(E * E - mass * mass, 0.0)))
        alt = {f"ke_{name}": self.ke(length_cm, name)
               for name in ("muon", "pion", "proton")}
        return dict(ke=ke, energy=E, p_mag=p, momentum=(p * d).astype(np.float32),
                    fourvec=np.array([E, *(p * d)], np.float32),
                    mass=mass, extra=alt)
