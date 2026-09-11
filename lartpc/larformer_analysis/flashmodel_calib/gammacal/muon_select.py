"""Calibration-muon candidates from a kp2 file + its nu_reco event group.

Reproduces the ntuple exporter's prong definitions (export_gen2ntuple.py) so
the selection matches what an ntuple-based cut would give, without needing the
ntuple: tracks = nu_reco particles with part_kind==0 (start = part_start_cm,
end = last polyline point, KE = part_energy - m(pred class), LArFormerPID =
LARFORMER_PDG[part_pred_class], IsSecondary from the attach vertex depth), plus
VERTEX-LESS kp2 particles not referenced by nu_reco (cls attr, start_cm/end_cm).

The candidate list is deliberately loose (muon class, length > min_len); every
protocol cut is a stored variable applied at fit time so cuts can be scanned
without rebuilding records.
"""
import numpy as np

from lartpc.larformer_reco.export.export_gen2ntuple import (
    LARFORMER_PDG, MASS, prong_rows)

TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])
BOUNDARY_MARGIN_CM = 10.0
MU_CLASS = 2


def on_boundary(pt, margin=BOUNDARY_MARGIN_CM):
    pt = np.asarray(pt, np.float64)
    if not np.isfinite(pt).all():
        return False
    return bool(np.any(np.abs(pt - TPC_LO) < margin)
                or np.any(np.abs(pt - TPC_HI) < margin))


def track_geometry(pts):
    """Principal-axis geometry of an instance's points: (end_a, end_b, length,
    linearity = 1 - lambda2/lambda1, rms transverse distance [cm]). Endpoints are
    the extreme points along the principal axis."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 3:
        return None
    c = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    proj = (pts - c) @ ax
    ia, ib = int(np.argmin(proj)), int(np.argmax(proj))
    perp = (pts - c) - np.outer(proj, ax)
    rms_perp = float(np.sqrt((perp ** 2).sum(1).mean()))
    lam = sv ** 2 / max(len(pts) - 1, 1)
    lin = float(1.0 - lam[1] / lam[0]) if lam[0] > 0 else 0.0
    return pts[ia], pts[ib], float(proj[ib] - proj[ia]), lin, rms_perp


def event_particles(kp, reco_group):
    """List of particle dicts for one event (reco + orphan kp2 instances).

    Every particle with a kp2 instance also carries the principal-axis geometry
    of its points (`lin`, `rms_perp`, `geo_len`); vertex-less instances whose
    end keypoint is missing get their endpoints from that geometry (the start
    keypoint is kept when it is finite and lies at one extreme)."""
    parts = []
    attached = set()
    npart = int(kp.attrs.get("n_particles", 0))
    slice_coords = (kp["slice/coord_cm"][()].astype(np.float32)
                    if "slice" in kp and "coord_cm" in kp["slice"] else None)

    def geom(inst):
        if slice_coords is None or inst < 0 or f"particle/{inst}" not in kp:
            return None
        pidx = kp[f"particle/{inst}/point_idx"][()].astype(np.int64)
        return track_geometry(slice_coords[pidx]) if len(pidx) >= 3 else None
    if reco_group is not None:
        d = prong_rows(reco_group)
        # prong_rows does not load part_length / part_true_ke; read them here
        for k in ("part_length", "part_true_ke"):
            if k in reco_group and k not in d:
                d[k] = reco_group[k][()]
        P = len(d.get("part_kind", []))
        for i in range(P):
            kind = int(d["part_kind"][i])
            inst = int(d["part_inst_idx"][i])
            if inst >= 0:
                attached.add(inst)
            cls = int(d["part_pred_class"][i])
            vloc = int(d["part_vtx"][i])
            e = float(d["part_energy"][i])
            ke = e - MASS.get(cls, 0.0) if kind == 0 else e
            st = np.asarray(d["part_start_cm"][i], np.float64)
            if kind == 0:
                poly = d["polys"][i]
                en = np.asarray(poly[-1], np.float64) if len(poly) else np.full(3, np.nan)
                length = float(d["part_length"][i]) if "part_length" in d else np.nan
                if not np.isfinite(length) and len(poly) > 1:
                    seg = np.diff(np.asarray(poly, np.float64), axis=0)
                    length = float(np.linalg.norm(seg, axis=1).sum())
            else:
                en = np.full(3, np.nan); length = np.nan
            lp = d.get("larpid_pid"); ls = d.get("larpid_scores")
            gm = geom(inst)
            parts.append(dict(
                lin=gm[3] if gm else np.nan, rms_perp=gm[4] if gm else np.nan,
                geo_len=gm[2] if gm else np.nan,
                inst=inst, orphan=False, kind=kind, cls=cls,
                pdg=LARFORMER_PDG.get(cls, 0), start=st, end=en, length=length,
                ke=ke, charge=float(d["part_charge"][i]),
                is_secondary=int(vloc >= 0 and d["vtx_depth"][vloc] > 0),
                vertexed=int(vloc >= 0), interaction=int(d["part_interaction"][i]),
                larpid_pid=int(lp[i]) if lp is not None else -1,
                larpid_mu=float(ls[i][2]) if ls is not None else np.nan,
                gt_trackid=int(d["part_gt_trackid"][i]) if "part_gt_trackid" in d else -1,
                true_ke=float(d["part_true_ke"][i]) if "part_true_ke" in d else np.nan))
    for inst in range(npart):
        if inst in attached or f"particle/{inst}" not in kp:
            continue
        g = kp[f"particle/{inst}"]
        cls = int(g.attrs.get("cls", 5))
        kind = 1 if cls in (0, 1) else 0
        st = np.asarray(g["start_cm"][()], np.float64)
        en = np.asarray(g["end_cm"][()], np.float64) if kind == 0 else np.full(3, np.nan)
        gm = geom(inst)
        if gm is not None and not (np.isfinite(st).all() and np.isfinite(en).all()):
            # endpoints from the principal axis; keep the kp start if it sits at
            # one extreme (within 10 cm), else use the geometric extremes
            a, b = gm[0], gm[1]
            if np.isfinite(st).all() and np.linalg.norm(st - b) < np.linalg.norm(st - a):
                a, b = b, a
            if not np.isfinite(st).all() or np.linalg.norm(st - a) > 10.0:
                st = a
            en = b
        length = float(np.linalg.norm(en - st)) if np.isfinite(en).all() else np.nan
        parts.append(dict(
            lin=gm[3] if gm else np.nan, rms_perp=gm[4] if gm else np.nan,
            geo_len=gm[2] if gm else np.nan,
            inst=inst, orphan=True, kind=kind, cls=cls, pdg=LARFORMER_PDG.get(cls, 0),
            start=st, end=en, length=length, ke=np.nan, charge=np.nan,
            is_secondary=-1, vertexed=0, interaction=-1, larpid_pid=-1,
            larpid_mu=np.nan, gt_trackid=int(g.attrs.get("gt_trackid", -1)),
            true_ke=np.nan))
    return parts


def cluster_particle(coords):
    """The whole flash-matched cluster as ONE candidate (flash-calib stream):
    endpoints/length/linearity from its points. inst=-1, orphan=2 marks it."""
    gm = track_geometry(coords)
    if gm is None:
        return None
    a, b, length, lin, rms = gm
    return dict(lin=lin, rms_perp=rms, geo_len=length, inst=-1, orphan=2, kind=0,
                cls=-1, pdg=13, start=np.asarray(a, np.float64), end=np.asarray(b, np.float64),
                length=float(length), ke=np.nan, charge=np.nan, is_secondary=-1,
                vertexed=0, interaction=-1, larpid_pid=-1, larpid_mu=np.nan,
                gt_trackid=-1, true_ke=np.nan)


def candidate_muons(parts, min_len=30.0, require_mu_class=True):
    """Loose candidates with the isolation variables of the event attached.
    Every protocol cut is applied later.

    require_mu_class=True  (production nu stream): segmenter muon-class tracks.
    require_mu_class=False (flash-calib stream): EVERY instance with a finite
        length above min_len, whatever its class -- the segmenter labels many
        cosmic clusters electron/photon; the fit then selects track-like
        geometry (`lin`, `rms_perp`) and/or the class."""
    out = []
    if require_mu_class:
        tracks = [p for p in parts if p["kind"] == 0]
        showers = [p for p in parts if p["kind"] == 1]
    else:
        tracks = list(parts)
        showers = []
    for p in tracks:
        if require_mu_class and p["pdg"] != 13:
            continue
        if not np.isfinite(p["length"]) or p["length"] <= min_len:
            continue
        others = [t for t in tracks if t is not p]
        other_ke = [t["ke"] for t in others if np.isfinite(t["ke"])]
        proton_ke = [t["ke"] for t in others
                     if t["pdg"] == 2212 and np.isfinite(t["ke"])]
        shower_e = [s["ke"] for s in showers if np.isfinite(s["ke"])]
        sb, eb = on_boundary(p["start"]), on_boundary(p["end"])
        bx = p["start"][0] if sb else (p["end"][0] if eb else np.nan)
        bxs = [p["start"][0] if sb else np.nan, p["end"][0] if eb else np.nan]
        q = dict(p)
        q.update(n_boundary=int(sb) + int(eb), boundary_end_x=float(bx),
                 boundary_x_min=float(np.nanmin(bxs)) if (sb or eb) else np.nan,
                 iso_track_ke_max=float(max(other_ke)) if other_ke else 0.0,
                 n_other_tracks=len(others),
                 n_other_tracks_orphan=sum(1 for t in others if t["orphan"]),
                 proton_ke_max=float(max(proton_ke)) if proton_ke else 0.0,
                 iso_shower_e_max=float(max(shower_e)) if shower_e else 0.0,
                 n_showers=len(showers))
        out.append(q)
    return out
