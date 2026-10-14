#!/usr/bin/env python3
"""
convert_larcv2_to_larformer_h5.py -- SBND larcv2 (SPINE/mlreco) files -> per-event
LArFormer HDF5 ("merged_sp"), the format read by pointcept.datasets.larformer and the
larformer_reco chain (see ../uboone_official/LARFORMER_DATAPREP.md for the MicroBooNE
version and ./README.md for the field-by-field analysis this implements).

    # inside the pointcept container, larcv2 set up (see run_convert.sh)
    python3 convert_larcv2_to_larformer_h5.py -i FILE.root [FILE2.root ...] -o OUTDIR \
        [--tag sbnd_mc] [--nentries N] [--drop-2plane]

Per event, entry_0/:
  triplet_data/   one row per sparse3d_reco voxel (ghost-included, like the MicroBooNE
                  triplet proposals):
     pos (N,3) voxel centre [cm]; pixval (N,3) hit integral on planes (U,V,Y) (0 = no hit);
     uwire/vwire/ywire nearest wire per plane of the point's TPC and tick (collection plane)
     from lartpc.detinfo; tpc; hasmatch (1 real / 0 ghost, pcluster_semantics_ghost != 5);
     trackid / pid / origin of the owning particle (0 for ghosts); ssnet_label (MicroBooNE
     0-9 classes rebuilt from SPINE shape + PDG); lm_score (1.0 dummy); edep (N,3) zeros +
     edep_mev (N,) true MeV on the voxel; SBND extras: reco_charge, rescaled_charge,
     hit_key (N,3), nhits, reco_chi2, reco_occupancy, interaction_id, is_dirt
  mc_particle_tree/  particle_corrected: trackid, parent_trackid, pid, origin, energy_mev
                  (KINETIC), start_pos (+ SCE-shifted start_pos_sce), daughters, nu_vertices,
                  plus shape, interaction_id, energy_deposit, contained, is_dirt, first/last_step
  mckeypoints/    larflow MCKeypointMaker rules (nu vertex, track start/end, shower/michel/delta)
                  with a visibility requirement (>= 2 real voxels within 3 cm)
  shower_fragments/ empty group (opened unconditionally by the dataset)
  flashes/        producer 0 = PMT flashes, 1 = X-ARAPUCA flashes; coincident tpc0/tpc1 flashes
                  (|dt| <= --flash-merge-us) are summed into one 312-channel vector; tpc_mask bits
  pmt_positions   (312,3) optical-detector centres; attrs n_pmts=312, pd_type list
  crthits/        x,y,z,errors,t_ns,pe,wall
  truth/neutrinos table (pdg, ccnc, mode, interaction_type, E [GeV], vertex, t, in_tpc,
                  lepton_pdg/lepton_energy from the GENIE primaries)
  attrs: run (1), subrun (crc32 of the input file name), event (entry), source_file, detector

Label choices (README decisions D1-D8, defaults):
  D1 2-plane space points are kept (nhits stored); --drop-2plane removes them.
  D3 origin: 1 = particles of a neutrino interaction whose vertex is inside the TPC active
     volume; 2 = cosmic AND dirt-neutrino particles (background), with is_dirt = 1 for the
     latter.  nu_vertices / mckeypoints nu vertices: in-TPC neutrinos only.
  D6 start_pos_sce: true start shifted by the SBND forward SCE offsets (lartpc.detinfo),
     only if --sce-keypoints (decided per sample by --check-sce); otherwise = start_pos.
  D7 beam flash: not chosen here; downstream picks the brightest producer-0 flash in a window
     (attrs flashes/beam_window_us = [0, 2.5] us as the MC suggestion).
"""
import argparse
import os
import sys
import zlib

import numpy as np
import h5py                                     # BEFORE larcv (HDF5 version clash)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

PDG_MASS_MEV = {11: 0.51099895, 13: 105.6583755, 15: 1776.86, 22: 0.0, 111: 134.9768,
                211: 139.57039, 321: 493.677, 130: 497.611, 310: 497.611, 2212: 938.27208816,
                2112: 939.56542052, 3122: 1115.683, 12: 0.0, 14: 0.0, 16: 0.0}
SHAPE_SHOWER, SHAPE_TRACK, SHAPE_MICHEL, SHAPE_DELTA, SHAPE_LE = 0, 1, 2, 3, 4
GHOST_SEM = 5
# larflow ssnet classes: 0 bg/ghost, 1 e, 2 gamma, 3 mu, 4 p(n), 5 pi/K, 6 michel, 7 delta, 8 LE, 9 other
KP_NU, KP_TRK_START, KP_TRK_END, KP_SHOWER, KP_MICHEL, KP_DELTA = range(6)
DELTA_PROCESSES = ("muIoni", "muBrems", "muPairProd", "eBrem", "muBrem")
MICHEL_PROCESSES = ("Decay", "muMinusCaptureAtRest")


def mass(pdg):
    a = abs(int(pdg))
    if a in PDG_MASS_MEV:
        return PDG_MASS_MEV[a]
    if a > 1000000000:                           # nucleus: A * 931.494 MeV
        return ((a // 10) % 1000) * 931.494
    return 0.0


def ssnet_class(shape, pdg):
    a = abs(int(pdg))
    if shape == SHAPE_MICHEL:
        return 6
    if shape == SHAPE_DELTA:
        return 7
    if shape == SHAPE_LE:
        return 8
    if a == 11:
        return 1
    if a == 22:
        return 2
    if a == 13:
        return 3
    if a in (2212, 2112):
        return 4
    if a in (211, 321):
        return 5
    return 9


# ----------------------------------------------------------------------------- larcv readers
def voxelset_arrays(vs):
    v = vs.as_vector()
    n = v.size()
    ids = np.fromiter((v[i].id() for i in range(n)), np.int64, n)
    vals = np.fromiter((v[i].value() for i in range(n)), np.float64, n)
    return ids, vals


def sparse(io, name):
    s = io.get_data("sparse3d", name)
    ids, vals = voxelset_arrays(s)
    return ids, vals, s.meta()


def aligned(ids_ref, ids, vals, fill):
    """values of (ids, vals) on the voxel set ids_ref (fill where absent)."""
    out = np.full(ids_ref.shape[0], fill, dtype=np.float64)
    if ids.size:
        order = np.argsort(ids)
        pos = np.searchsorted(ids[order], ids_ref)
        pos = np.clip(pos, 0, ids.size - 1)
        hit = ids[order][pos] == ids_ref
        out[hit] = vals[order][pos[hit]]
    return out


def voxel_xyz(meta, ids):
    nx, ny = meta.num_voxel_x(), meta.num_voxel_y()
    ix, iy, iz = ids % nx, (ids // nx) % ny, ids // (nx * ny)
    return np.stack([meta.min_x() + (ix + 0.5) * meta.size_voxel_x(),
                     meta.min_y() + (iy + 0.5) * meta.size_voxel_y(),
                     meta.min_z() + (iz + 0.5) * meta.size_voxel_z()], 1)


def vtx(v):
    return np.array([v.x(), v.y(), v.z()], np.float64)


# ----------------------------------------------------------------------------- converter
class Converter:
    def __init__(self, args):
        from lartpc.detinfo import load
        self.a = args
        self.di = load("sbnd", args.detinfo_version)
        g = self.di.geometry
        self.av_lo, self.av_hi = g.DetectorBoundingBox(active=True).min, g.DetectorBoundingBox(active=True).max
        self.n_opch = g.NOpChannels()
        self.pd_types = list(self.di.conventions.opdet_types())
        self.opdet_pos = g.OpDetPositions().astype(np.float32)
        aux = g.AuxDetTable()
        self.crt_centers = aux["center"]
        self.crt_walls = self.di.conventions.crt_walls()
        self.n_written = self.n_skipped = 0

    def in_av(self, p):
        p = np.asarray(p, np.float64)
        return bool(np.all(p >= self.av_lo) and np.all(p <= self.av_hi))

    def sce_shift(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if not self.a.sce_keypoints or len(pts) == 0:
            return pts
        return pts + self.di.sce.GetPosOffsets(pts)

    # ------------------------------------------------------------------ event
    def event(self, io, entry, src):
        from larcv import larcv  # noqa: F401  (already imported by caller)
        a = self.a
        r_ids, r_q, meta = sparse(io, "reco")
        if r_ids.size == 0:
            return None
        nh = aligned(r_ids, *sparse(io, "reco_nhits")[:2], 0).astype(np.int8)
        keep = np.ones(r_ids.size, bool) if not a.drop_2plane else nh >= 3
        r_ids, r_q, nh = r_ids[keep], r_q[keep], nh[keep]
        pos = voxel_xyz(meta, r_ids).astype(np.float32)
        N = r_ids.size

        hq = np.stack([aligned(r_ids, *sparse(io, f"reco_hit_charge{k}")[:2], -1) for k in range(3)], 1)
        hk = np.stack([aligned(r_ids, *sparse(io, f"reco_hit_key{k}")[:2], -1) for k in range(3)], 1)
        chi2 = aligned(r_ids, *sparse(io, "reco_chi2")[:2], np.nan)
        occ = aligned(r_ids, *sparse(io, "reco_occupancy")[:2], 0)
        resc = aligned(r_ids, *sparse(io, "reco_rescaled")[:2], 0)
        sem_g = aligned(r_ids, *sparse(io, "pcluster_semantics_ghost")[:2], GHOST_SEM).astype(np.int64)
        owner = aligned(r_ids, *sparse(io, "pcluster_index")[:2], -1).astype(np.int64)
        emev = aligned(r_ids, *sparse(io, "pcluster")[:2], 0)
        real = sem_g != GHOST_SEM

        # ---- truth: particles + neutrinos
        pv = io.get_data("particle", "corrected").as_vector()
        P = pv.size()
        nus = io.get_data("neutrino", "mpv").as_vector()
        nu_v = [vtx(nus[i].position()) for i in range(nus.size())]
        nu_in = [self.in_av(v) for v in nu_v]
        parts = []
        for k in range(P):
            p = pv[k]
            tid = int(p.track_id())
            gen = tid // 10_000_000                     # 1 = neutrino generator, 2 = CORSIKA
            anc = vtx(p.ancestor_position())
            nu_match = [i for i, v in enumerate(nu_v) if np.linalg.norm(v - anc) < 0.05]
            is_nu = gen == 1
            in_tpc_nu = is_nu and bool(nu_match) and nu_in[nu_match[0]]
            fs, ls = vtx(p.first_step()), vtx(p.last_step())
            parts.append(dict(
                idx=k, tid=tid, pdg=int(p.pdg_code()), shape=int(p.shape()),
                parent_tid=int(p.parent_track_id()), anc_pdg=int(p.ancestor_pdg_code()),
                parent_pdg=int(p.parent_pdg_code()), process=str(p.creation_process()),
                E=float(p.energy_init()), Edep=float(p.energy_deposit()), start=vtx(p.position()),
                t=float(p.t()), first=fs, last=ls, nvox=int(p.num_voxels()),
                iid=int(p.interaction_id()), group=int(p.group_id()),
                children=[int(c) for c in p.children_id()],
                origin=1 if in_tpc_nu else 2, is_dirt=int(is_nu and not in_tpc_nu),
                nu_index=nu_match[0] if nu_match else -1,
                contained=int(fs[0] > -1e6 and self.in_av(fs) and self.in_av(ls))))
        own_ok = real & (owner >= 0) & (owner < P)
        sp_tid = np.zeros(N, np.int64)
        sp_pid = np.zeros(N, np.int64)
        sp_org = np.zeros(N, np.int32)
        sp_ss = np.zeros(N, np.int32)
        sp_iid = np.full(N, -1, np.int64)
        sp_dirt = np.zeros(N, np.int8)
        lut = np.array([[q["tid"], q["pdg"], q["origin"], q["iid"], q["is_dirt"]] for q in parts],
                       np.int64).reshape(-1, 5)
        if own_ok.any():
            o = owner[own_ok]
            sp_tid[own_ok], sp_pid[own_ok], sp_org[own_ok] = lut[o, 0], lut[o, 1], lut[o, 2]
            sp_iid[own_ok], sp_dirt[own_ok] = lut[o, 3], lut[o, 4]
            sem = sem_g[own_ok]
            sp_ss[own_ok] = [ssnet_class(s if s != SHAPE_TRACK else parts[i]["shape"], parts[i]["pdg"])
                             for s, i in zip(sem, o)]

        # ---- wires / ticks from detinfo
        proj = self.di.project(pos)
        wires = proj["nearest_wire"]
        tick = np.rint(np.nan_to_num(proj["tick"][:, 2], nan=-1)).astype(np.int32)

        # ---- keypoints
        kp = self.keypoints(parts, nu_v, nu_in, pos[real])

        return dict(pos=pos, hq=hq, hk=hk, nh=nh, chi2=chi2, occ=occ, resc=resc, rq=r_q, real=real,
                    tid=sp_tid, pid=sp_pid, org=sp_org, ss=sp_ss, iid=sp_iid, dirt=sp_dirt, emev=emev,
                    wires=wires, tick=tick, tpc=proj["tpc"], parts=parts, nus=nus, nu_v=nu_v,
                    nu_in=nu_in, kp=kp)

    def keypoints(self, parts, nu_v, nu_in, real_pos):
        """larflow MCKeypointMaker rules on larcv2 particles (see module doc)."""
        from scipy.spatial import cKDTree
        rows = []
        for i, v in enumerate(nu_v):
            if nu_in[i]:
                rows.append((v, v, KP_NU, 0, 0))
        for q in parts:
            a = abs(q["pdg"])
            if q["first"][0] < -1e6:                      # never deposited
                continue
            if a in (11, 22):
                # shower roots only (larflow uses mcshower objects = shower roots)
                if abs(q["parent_pdg"]) in (11, 22) and q["parent_tid"] != q["tid"]:
                    continue
                mu_rel = abs(q["parent_pdg"]) == 13 or abs(q["anc_pdg"]) == 13
                if mu_rel:
                    t = KP_MICHEL if q["process"] in MICHEL_PROCESSES else KP_DELTA
                elif q["process"] in DELTA_PROCESSES:
                    t = KP_DELTA
                else:
                    t = KP_SHOWER
                rows.append((q["first"], q["start"], t, q["tid"], q["pdg"]))
            elif a != 2112 and a < 1000000000:
                if np.linalg.norm(q["last"] - q["first"]) < 3.0:
                    continue
                rows.append((q["first"], q["first"], KP_TRK_START, q["tid"], q["pdg"]))
                rows.append((q["last"], q["last"], KP_TRK_END, q["tid"], q["pdg"]))
        if not rows:
            return dict(pos=np.zeros((0, 3), np.float32), startpos=np.zeros((0, 3), np.float32),
                        kptype=np.zeros(0, np.int32), trackid=np.zeros(0, np.int64), pid=np.zeros(0, np.int32))
        p = self.sce_shift(np.array([r[0] for r in rows]))
        sp = self.sce_shift(np.array([r[1] for r in rows]))
        t = np.array([r[2] for r in rows], np.int32)
        if len(real_pos):
            tree = cKDTree(real_pos)
            nvis = np.array([len(x) for x in tree.query_ball_point(p, r=3.0)])
        else:
            nvis = np.zeros(len(rows), int)
        vis = nvis >= 2
        return dict(pos=p[vis].astype(np.float32), startpos=sp[vis].astype(np.float32), kptype=t[vis],
                    trackid=np.array([r[3] for r in rows], np.int64)[vis],
                    pid=np.array([r[4] for r in rows], np.int32)[vis])

    # ------------------------------------------------------------------ flashes / CRT
    def flashes(self, io):
        out = []
        for pid, prods in ((0, ("tpc0", "tpc1")), (1, ("tpc0xa", "tpc1xa"))):
            fl = []
            for t, prod in enumerate(prods):
                fv = io.get_data("opflash", prod).as_vector()
                for i in range(fv.size()):
                    f = fv[i]
                    pe = np.zeros(self.n_opch, np.float32)
                    v = np.asarray(list(f.PEPerOpDet()), np.float32)
                    pe[: min(self.n_opch, v.size)] = v[: self.n_opch]
                    fl.append(dict(t=float(f.time()), pe=pe, tpc=t, y=f.yCenter(), z=f.zCenter(), idx=i))
            fl.sort(key=lambda d: d["t"])
            used = [False] * len(fl)
            for i, f in enumerate(fl):
                if used[i]:
                    continue
                used[i] = True
                group = [f]
                for j in range(i + 1, len(fl)):
                    if fl[j]["t"] - f["t"] > self.a.flash_merge_us:
                        break
                    if not used[j] and fl[j]["tpc"] != f["tpc"] and all(g["tpc"] != fl[j]["tpc"] for g in group):
                        used[j] = True
                        group.append(fl[j])
                pe = np.sum([g["pe"] for g in group], 0)
                tot = float(pe.sum())
                w = np.array([g["pe"].sum() for g in group], np.float64)
                out.append(dict(pe=pe, total_pe=tot, time_us=float(np.average([g["t"] for g in group], weights=w + 1e-9)),
                                producer_id=pid, tpc_mask=int(sum(1 << g["tpc"] for g in group)),
                                y_center=float(np.average([g["y"] for g in group], weights=w + 1e-9)),
                                z_center=float(np.average([g["z"] for g in group], weights=w + 1e-9))))
        return out

    def crt(self, io):
        cv = io.get_data("crthit", "crthit").as_vector()
        n = cv.size()
        if n == 0:
            return None
        xyz = np.array([[cv[i].x_pos(), cv[i].y_pos(), cv[i].z_pos()] for i in range(n)], np.float32)
        err = np.array([[cv[i].x_err(), cv[i].y_err(), cv[i].z_err()] for i in range(n)], np.float32)
        d = np.linalg.norm(xyz[:, None, :] - self.crt_centers[None, :, :], axis=-1)
        wall = self.crt_walls[np.argmin(d, axis=1)]
        return dict(pos=xyz, err=err, t_ns=np.array([cv[i].ts1_ns() for i in range(n)], np.float64),
                    pe=np.array([cv[i].peshit() for i in range(n)], np.float32),
                    wall=np.array([str(w) for w in wall], dtype=h5py.string_dtype()))

    # ------------------------------------------------------------------ write
    def write(self, path, ev, flashes, crt, run, subrun, event, src, entry):
        with h5py.File(path, "w") as f:
            e = f.create_group("entry_0")
            for k, v in dict(run=run, subrun=subrun, event=event, source_file=os.path.basename(src),
                             source_entry=entry, detector="sbnd",
                             detinfo=os.path.basename(self.di.path)).items():
                e.attrs[k] = v
            N = ev["pos"].shape[0]
            td = e.create_group("triplet_data")
            gz = dict(compression="gzip", compression_opts=4, shuffle=True)
            td.create_dataset("pos", data=ev["pos"], **gz)
            td.create_dataset("pixval", data=np.clip(ev["hq"], 0, None).astype(np.float32), **gz)
            for k, name in enumerate(("uwire", "vwire", "ywire")):
                td.create_dataset(name, data=ev["wires"][:, k].astype(np.int32), **gz)
            td.create_dataset("tick", data=ev["tick"], **gz)
            td.create_dataset("tpc", data=ev["tpc"].astype(np.int8), **gz)
            td.create_dataset("hasmatch", data=ev["real"].astype(np.int32), **gz)
            td.create_dataset("trackid", data=ev["tid"], **gz)
            td.create_dataset("pid", data=ev["pid"].astype(np.int32), **gz)
            td.create_dataset("origin", data=ev["org"], **gz)
            td.create_dataset("ssnet_label", data=ev["ss"], **gz)
            td.create_dataset("lm_score", data=np.ones(N, np.float32), **gz)
            td.create_dataset("edep", data=np.zeros((N, 3), np.float32), **gz)
            td.create_dataset("edep_mev", data=ev["emev"].astype(np.float32), **gz)
            td.create_dataset("reco_charge", data=ev["rq"].astype(np.float32), **gz)
            td.create_dataset("rescaled_charge", data=ev["resc"].astype(np.float32), **gz)
            td.create_dataset("hit_key", data=ev["hk"].astype(np.int32), **gz)
            td.create_dataset("nhits", data=ev["nh"], **gz)
            td.create_dataset("reco_chi2", data=ev["chi2"].astype(np.float32), **gz)
            td.create_dataset("reco_occupancy", data=ev["occ"].astype(np.int8), **gz)
            td.create_dataset("interaction_id", data=ev["iid"], **gz)
            td.create_dataset("is_dirt", data=ev["dirt"], **gz)
            td.attrs["pixval_units"] = "SBND hit integral (ADC x tick), whole hit on every voxel built from it"
            td.attrs["wire_tick_source"] = "lartpc.detinfo nearest wire / collection-plane tick of the point's TPC"

            parts = ev["parts"]
            mp = e.create_group("mc_particle_tree")
            tid2 = {q["tid"]: q for q in parts}
            starts = np.array([q["start"] for q in parts], np.float32).reshape(-1, 3)
            dt, ds, nd = [], [], []
            for q in parts:
                ds.append(len(dt))
                kids = [parts[c]["tid"] for c in q["children"] if 0 <= c < len(parts) and parts[c]["tid"] != q["tid"]]
                nd.append(len(kids))
                dt.extend(kids)
            cols = dict(
                trackid=np.array([q["tid"] for q in parts], np.int64),
                parent_trackid=np.array([q["parent_tid"] if q["parent_tid"] in tid2 else -1 for q in parts], np.int64),
                pid=np.array([q["pdg"] for q in parts], np.int32),
                origin=np.array([q["origin"] for q in parts], np.int32),
                process_code=np.full(len(parts), -1, np.int32),
                energy_mev=np.array([max(q["E"] - mass(q["pdg"]), 0.0) for q in parts], np.float32),
                start_pos=starts, start_pos_sce=self.sce_shift(starts).astype(np.float32),
                num_daughters=np.array(nd, np.int32), daughter_start_indices=np.array(ds, np.int32),
                daughter_trackids=np.array(dt, np.int64),
                shape=np.array([q["shape"] for q in parts], np.int32),
                interaction_id=np.array([q["iid"] for q in parts], np.int64),
                energy_deposit=np.array([q["Edep"] for q in parts], np.float32),
                contained=np.array([q["contained"] for q in parts], np.int8),
                is_dirt=np.array([q["is_dirt"] for q in parts], np.int8),
                first_step=np.array([q["first"] for q in parts], np.float32).reshape(-1, 3),
                last_step=np.array([q["last"] for q in parts], np.float32).reshape(-1, 3),
                creation_process=np.array([q["process"] for q in parts], dtype=h5py.string_dtype()))
            for k, v in cols.items():
                mp.create_dataset(k, data=v)
            nvi = [v for v, ok in zip(ev["nu_v"], ev["nu_in"]) if ok]
            mp.create_dataset("nu_vertices", data=np.array(nvi, np.float32).reshape(-1, 3))
            mp.attrs["energy_mev"] = "kinetic energy"
            mp.attrs["start_pos_sce"] = ("start_pos + SBND forward SCE offsets" if self.a.sce_keypoints
                                         else "= start_pos (no SCE shift)")

            kp = ev["kp"]
            mk = e.create_group("mckeypoints")
            for k in ("pos", "startpos", "kptype", "trackid", "pid"):
                mk.create_dataset(k, data=kp[k])
            mk.create_dataset("imgcoord", data=np.zeros((len(kp["kptype"]), 0), np.int32))

            sf = e.create_group("shower_fragments")
            sf.attrs["num_fragments"] = np.int32(0)
            for k in ("istrunk", "nu_vertex_is_visible", "pid", "pointindices_counts", "trackid", "type"):
                sf.create_dataset(k, data=np.int32(0))
            sf.create_dataset("pointindices_flat", data=np.zeros(0, np.int64))
            for k in ("originpt", "pret0shiftedoriginpt", "startpt"):
                sf.create_dataset(k, data=np.zeros((0, 1), np.float32))

            F = len(flashes)
            fl = e.create_group("flashes")
            fl.attrs["num_flashes"] = F
            fl.attrs["producers"] = "0 = PMT (opflashtpc0 + opflashtpc1), 1 = X-ARAPUCA (opflashtpc{0,1}xarapuca)"
            fl.attrs["merge_window_us"] = float(self.a.flash_merge_us)
            fl.attrs["beam_window_us"] = np.array([0.0, 2.5])
            col = lambda k, dt: np.array([x[k] for x in flashes], dt) if F else np.zeros(0, dt)
            fl.create_dataset("pe", data=(np.stack([x["pe"] for x in flashes]) if F
                                          else np.zeros((0, self.n_opch), np.float32)), **gz)
            for k, dt in (("total_pe", np.float32), ("time_us", np.float32), ("producer_id", np.int32),
                          ("tpc_mask", np.int32), ("y_center", np.float32), ("z_center", np.float32)):
                fl.create_dataset(k, data=col(k, dt))
            fl.create_dataset("flash_index", data=np.arange(F, dtype=np.int32))
            fl.create_dataset("tpc_tick", data=np.zeros(F, np.float32))
            e.create_dataset("pmt_positions", data=self.opdet_pos)
            e.attrs["n_pmts"] = self.n_opch
            e.attrs["usec_per_tick"] = float(self.di.clocks.TPCClock().TickPeriod)
            e.attrs["trigger_tick"] = int(round(-self.di.clocks.TriggerOffsetTPC() / self.di.clocks.TPCClock().TickPeriod))
            e.create_dataset("pd_type", data=np.array(self.pd_types, dtype=h5py.string_dtype()))

            if crt is not None:
                cg = e.create_group("crthits")
                for k, v in crt.items():
                    cg.create_dataset(k, data=v)

            tg = e.create_group("truth").create_group("neutrinos")
            nus = ev["nus"]
            rows = []
            for i in range(nus.size()):
                n = nus[i]
                v = ev["nu_v"][i]
                lep = [q for q in ev["parts"] if np.linalg.norm(q["start"] - v) < 0.05 and abs(q["pdg"]) in (11, 13, 15, 12, 14, 16)]
                lep = max(lep, key=lambda q: q["E"]) if lep else None
                rows.append((n.pdg_code(), n.current_type(), n.interaction_mode(), n.interaction_type(),
                             n.energy_init(), v[0], v[1], v[2], n.t(), int(ev["nu_in"][i]),
                             lep["pdg"] if lep else 0, lep["E"] if lep else -1.0))
            names = ("pdg", "ccnc", "mode", "interaction_type", "energy_gev", "vtx_x", "vtx_y", "vtx_z",
                     "t_ns", "in_tpc", "lepton_pdg", "lepton_energy_mev")
            arr = np.array(rows, np.float64).reshape(-1, len(names))
            for j, nm in enumerate(names):
                tg.create_dataset(nm, data=arr[:, j])
            tg.attrs["note"] = "neutrino_mpv; ccnc 0=CC 1=NC; lepton from particle_corrected at the vertex"

    # ------------------------------------------------------------------ driver
    def run(self, files):
        from larcv import larcv
        os.makedirs(self.a.output_dir, exist_ok=True)
        for src in files:
            io = larcv.IOManager(larcv.IOManager.kREAD)
            io.add_in_file(src)
            io.set_verbosity(2)
            io.initialize()
            n = io.get_n_entries()
            end = n if self.a.nentries < 0 else min(n, self.a.start_entry + self.a.nentries)
            subrun = zlib.crc32(os.path.basename(src).encode()) & 0x7FFFFFFF
            stem = os.path.splitext(os.path.basename(src))[0]
            for entry in range(self.a.start_entry, end):
                io.read_entry(entry)
                ev = self.event(io, entry, src)
                if ev is None:
                    self.n_skipped += 1
                    print(f"  [{entry}] no reco voxels; skipped")
                    continue
                fls = self.flashes(io)
                crt = self.crt(io)
                out = os.path.join(self.a.output_dir,
                                   f"{self.a.out_prefix}_{self.a.tag}_{stem[:24]}_entry{entry:06d}.h5")
                self.write(out, ev, fls, crt, self.a.run, subrun, entry, src, entry)
                self.n_written += 1
                print(f"  [{entry}] -> {os.path.basename(out)}  ({ev['pos'].shape[0]} points, "
                      f"{int(ev['real'].sum())} real, {len(ev['kp']['kptype'])} keypoints, "
                      f"{sum(ev['nu_in'])}/{len(ev['nu_in'])} nu in TPC, {len(fls)} flashes)")
            io.finalize()
        print(f"done: {self.n_written} written, {self.n_skipped} skipped")


def check_sce(conv, files, n_events=5):
    """Is particle truth in the true frame (needs the forward SCE shift to sit on the reco
    voxels)? Median distance from track start/end keypoints to the particle's OWN real voxels,
    with and without the shift."""
    from larcv import larcv
    from scipy.spatial import cKDTree
    io = larcv.IOManager(larcv.IOManager.kREAD)
    for s in files:
        io.add_in_file(s)
    io.set_verbosity(2)
    io.initialize()
    d0, d1 = [], []
    for e in range(min(n_events, io.get_n_entries())):
        io.read_entry(e)
        ids, _, meta = sparse(io, "pcluster_index")
        own = aligned(ids, *sparse(io, "pcluster_index")[:2], -1).astype(int)
        pos = voxel_xyz(meta, ids)
        pv = io.get_data("particle", "corrected").as_vector()
        for k in range(pv.size()):
            p = pv[k]
            if abs(p.pdg_code()) not in (13, 2212, 211) or p.first_step().x() < -1e6:
                continue
            m = own == k
            if m.sum() < 20:
                continue
            tree = cKDTree(pos[m])
            ends = np.array([vtx(p.first_step()), vtx(p.last_step())])
            if not (conv.in_av(ends[0]) and conv.in_av(ends[1])):
                continue
            d0.extend(tree.query(ends)[0])
            d1.extend(tree.query(ends + conv.di.sce.GetPosOffsets(ends))[0])
    d0, d1 = np.array(d0), np.array(d1)
    print(f"SCE check on {len(d0)} contained track ends: median distance to own voxels "
          f"without shift {np.median(d0):.2f} cm, with forward SCE shift {np.median(d1):.2f} cm "
          f"(68%: {np.percentile(d0, 68):.2f} / {np.percentile(d1, 68):.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", nargs="+", required=True)
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--tag", default="sbnd_mc")
    ap.add_argument("--out-prefix", default="merged")
    ap.add_argument("-n", "--nentries", type=int, default=-1)
    ap.add_argument("-e", "--start-entry", type=int, default=0)
    ap.add_argument("--run", type=int, default=1, help="run number written to the event attrs")
    ap.add_argument("--drop-2plane", action="store_true", help="drop space points built from 2 planes")
    ap.add_argument("--sce-keypoints", action="store_true",
                    help="shift truth positions (keypoints, start_pos_sce) by the SBND forward SCE offsets")
    ap.add_argument("--check-sce", action="store_true", help="only run the SCE-frame check and exit")
    ap.add_argument("--flash-merge-us", type=float, default=0.25,
                    help="merge tpc0/tpc1 flashes closer than this into one 312-channel flash")
    ap.add_argument("--detinfo-version", default=None)
    a = ap.parse_args()
    conv = Converter(a)
    from larcv import larcv  # noqa: F401  -- after h5py + detinfo
    if a.check_sce:
        check_sce(conv, a.input)
        return
    conv.run(a.input)


if __name__ == "__main__":
    main()
