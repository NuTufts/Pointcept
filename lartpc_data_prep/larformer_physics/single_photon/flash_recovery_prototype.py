"""
Phase-4d: flash-recovery prototype for single photons mislabeled into a cosmic slice.

For each photon whose dominant predicted slice is COSMIC but is its OWN slice
(category own_cosmic_slice from analyze_photon_slice.py), we ask: would matching the
in-time BEAM flash to each slice's PhotonLib-predicted PMT pattern recover it?

Per event:
  - in-time beam flash = producer_id==0 flash with max total_pe; observed pe_obs(32,), t0.
  - per slice (post/pred_query) with >= --min-pts points: predict PE(32,) via PhotonLib
    (predict_many_slices_pe, gamma_beam=5.25 from the gamma_tune summary), then
    Neyman chi2 vs pe_obs.
  - RECOVERED if the photon's slice has the lowest chi2 (rank 1) among slices.

Charge: raw Y-plane ADC with U/V fallback (select_charge_y_with_uv_fallback_np),
mapped from merged_sp triplet_data to the slicer post points by a fine voxel hash.

Run inside the pointcept container (needs torch + photonlib):
  PYTHONPATH=<pointcept> python3 flash_recovery_prototype.py \
      --pred-dir <stage3pred> --merged-dir <merged_sp> \
      --slice-csv workdir_scale/photon_slice.csv --gamma-beam 5.25 --out workdir_scale/flash_recovery.csv
"""
import argparse
import csv
import glob
import os
import sys
import h5py
import numpy as np
from collections import Counter, defaultdict

REPO = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept"
sys.path.insert(0, REPO)
from lartpc_data_prep.larformer_analysis.lib.flash_predict import (   # noqa: E402
    predict_many_slices_pe, select_charge_y_with_uv_fallback_np)
from lartpc_data_prep.larformer_analysis.lib.flash_chi2 import neyman_chi2  # noqa: E402

PHOTON_PDG = 22
VOX = 1.0


def vkeys(pts):
    return [tuple(v) for v in np.floor(pts / VOX).astype(np.int64)]


def photon_frag_idx(e):
    out = {}
    sf = e["shower_fragments"]
    pid = sf["pid"][:]; tid = sf["trackid"][:]; istr = sf["istrunk"][:]
    cnt = sf["pointindices_counts"][:]; flat = sf["pointindices_flat"][:]
    offs = np.concatenate([[0], np.cumsum(cnt)])
    for t in np.unique(tid[pid == PHOTON_PDG]):
        fr = np.where((tid == t) & (pid == PHOTON_PDG))[0]
        out[int(t)] = np.unique(np.concatenate([flat[offs[i]:offs[i + 1]] for i in fr]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--slice-csv", required=True)
    ap.add_argument("--gamma-beam", type=float, default=5.25)
    ap.add_argument("--min-pts", type=int, default=20)
    ap.add_argument("--only-1g0X", action="store_true")
    ap.add_argument("--out", default="workdir_scale/flash_recovery.csv")
    args = ap.parse_args()

    # own-cosmic-slice photons to test: (run,subrun,event) -> trackid
    targets = {}
    for r in csv.DictReader(open(args.slice_csv)):
        if r["category"] != "own_cosmic_slice":
            continue
        if args.only_1g0X and int(r["is_1g0X"]) != 1:
            continue
        targets[(int(r["run"]), int(r["subrun"]), int(r["event"]))] = (int(r["trackid"]), int(r["is_1g0X"]))
    print("own_cosmic_slice photons to test:", len(targets))

    merged = {os.path.basename(p): p
              for p in glob.glob(os.path.join(args.merged_dir, "**", "merged_*entry*.h5"), recursive=True)}

    rows = []
    for pf in sorted(glob.glob(os.path.join(args.pred_dir, "**", "stage3pred_*.h5"), recursive=True)):
        with h5py.File(pf, "r") as ph5:
            a = ph5.attrs
            key = (int(a.get("meta_run", -1)), int(a.get("meta_subrun", -1)), int(a.get("meta_event", -1)))
            if key not in targets:
                continue
            tid, is1g = targets[key]
            post = ph5["post"]["coord"][:]; pq = ph5["post"]["pred_query"][:]
        base = os.path.basename(pf)[len("stage3pred_"):]
        mp = merged.get(base)
        if mp is None:
            continue
        with h5py.File(mp, "r") as mh5:
            e = mh5["entry_0"]
            pos = e["triplet_data"]["pos"][:]
            charge = select_charge_y_with_uv_fallback_np(e["triplet_data"]["pixval"][:])
            frags = photon_frag_idx(e)
            fl = e["flashes"]
            f_pe = fl["pe"][:]; f_pid = fl["producer_id"][:]; f_tpe = fl["total_pe"][:]
            f_t = fl["time_us"][:]
        # in-time beam flash
        beam = np.where(f_pid == 0)[0]
        if len(beam) == 0:
            continue
        bi = beam[np.argmax(f_tpe[beam])]
        pe_obs = f_pe[bi]; t0 = float(f_t[bi])

        # voxel -> charge (mean) from merged
        vox2q = {}
        for k, q in zip(vkeys(pos), charge):
            vox2q.setdefault(k, []).append(q)
        vox2charge = {k: float(np.mean(v)) for k, v in vox2q.items()}

        # photon's dominant slice
        ph_keys = vkeys(pos[frags[tid]])
        # map photon points -> post slice via post voxel hash
        postvox = {}
        for kk, q in zip(vkeys(post), pq):
            postvox.setdefault(kk, []).append(int(q))
        postvox = {kk: Counter(v).most_common(1)[0][0] for kk, v in postvox.items()}
        ph_slices = [postvox[kk] for kk in ph_keys if kk in postvox]
        if not ph_slices:
            continue
        photon_slice = Counter(ph_slices).most_common(1)[0][0]

        # build per-slice points + charge for prediction (slices with >= min-pts)
        slice_counts = Counter(int(q) for q in pq)
        keep_slices = [s for s, c in slice_counts.items() if c >= args.min_pts]
        if photon_slice not in keep_slices:
            keep_slices.append(photon_slice)
        sid_map = {s: i for i, s in enumerate(keep_slices)}
        sel = np.array([int(q) in sid_map for q in pq])
        spos = post[sel]
        scharge = np.array([vox2charge.get(kk, 0.0) for kk in vkeys(spos)], dtype=np.float32)
        scid = np.array([sid_map[int(q)] for q in pq[sel]], dtype=np.int64)

        pe_pred = predict_many_slices_pe(
            spos.astype(np.float32), scharge, scid, len(keep_slices),
            t0, producer_id=0, gamma_by_producer=(args.gamma_beam, args.gamma_beam))
        chi2 = np.array([neyman_chi2(pe_pred[i], pe_obs) for i in range(len(keep_slices))])
        order = np.argsort(np.where(np.isnan(chi2), np.inf, chi2))
        ranked = [keep_slices[i] for i in order]
        rank = ranked.index(photon_slice) + 1
        rows.append((key[0], key[1], key[2], tid, is1g, len(keep_slices), rank,
                     float(chi2[sid_map[photon_slice]]),
                     float(np.nanmin(chi2))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("run,subrun,event,trackid,is_1g0X,n_slices,photon_slice_rank,"
                 "photon_chi2,best_chi2\n")
        for r in rows:
            fh.write("%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f\n" % r)

    n = len(rows)
    r1 = sum(1 for r in rows if r[6] == 1)
    r3 = sum(1 for r in rows if r[6] <= 3)
    g = [r for r in rows if r[4] == 1]
    g1 = sum(1 for r in g if r[6] == 1)
    print("=" * 60)
    print("flash recovery (gamma_beam=%.2f, min_pts=%d)" % (args.gamma_beam, args.min_pts))
    print("own-cosmic-slice photons tested : %d" % n)
    print("  photon slice = best flash match (rank 1) : %d (%.1f%%)" % (r1, 100.0 * r1 / max(n, 1)))
    print("  photon slice in top-3                    : %d (%.1f%%)" % (r3, 100.0 * r3 / max(n, 1)))
    print("  [1g0X subset] rank-1 : %d / %d (%.1f%%)" % (g1, len(g), 100.0 * g1 / max(len(g), 1)))
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
