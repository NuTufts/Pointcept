"""
Phase-4c: WHERE does the cascade lose a true photon? Slicer-level diagnosis.

For each visible true photon we find which predicted SLICE its spacepoints landed
in and that slice's class, to attribute photon loss to the slicer (event-slicer)
stage rather than the particle segmenter:

  category:
    nu_slice         photon's dominant predicted slice is class NU (0).
                     -> selectable; any miss is downstream (particle seg / keep-mask).
    own_cosmic_slice photon's slice is COSMIC(1)/no_object(2) BUT the photon makes up
                     most of that slice (frac >= --own-frac). I.e. the photon got its
                     OWN slice, just mislabeled cosmic -> RECOVERABLE by matching the
                     in-time beam flash to this slice's predicted PMT pattern.
    merged_cosmic    photon's slice is cosmic and the photon is a minority of it
                     (merged in with cosmic activity, e.g. a cosmic muon) -> harder.
    deghosted/lost   few/none of the photon's points survived into the slicer points.

Photon class id (slicer): nu=0, cosmic=1, no_object=2.

Matching: the photon's merged_sp triplet_data points (cm) are looked up in the
slicer `post/coord` (cm) via a fine voxel hash to get each point's slice
(post/pred_query); the slice class is queries/class_argmax[slice].

Run inside the pointcept container:
  python3 analyze_photon_slice.py --pred-dir <stage3pred> --merged-dir <merged_sp> \
      --truth-csv workdir_scale/visible_photon_events.csv --out workdir_scale/photon_slice.csv
"""
import argparse
import csv
import glob
import os
import h5py
import numpy as np
from collections import Counter

PHOTON_PDG = 22
NU_SLICE_CLASS = 0
VOX = 1.0   # cm, for matching merged photon points to slicer post points


def vkey(pts):
    return [tuple(v) for v in np.floor(pts / VOX).astype(np.int64)]


def photon_frag_indices(e):
    """trackid -> (unique triplet indices, nSP_trunk) for nu photons with a trunk."""
    out = {}
    if "shower_fragments" not in e or int(e["shower_fragments"].attrs.get("num_fragments", 0)) == 0:
        return out
    sf = e["shower_fragments"]
    pid = sf["pid"][:]; tid = sf["trackid"][:]; istrunk = sf["istrunk"][:]
    cnt = sf["pointindices_counts"][:]; flat = sf["pointindices_flat"][:]
    offs = np.concatenate([[0], np.cumsum(cnt)])
    for t in np.unique(tid[pid == PHOTON_PDG]):
        fr = np.where((tid == t) & (pid == PHOTON_PDG))[0]
        idx = np.unique(np.concatenate([flat[offs[i]:offs[i + 1]] for i in fr]))
        trunk = fr[istrunk[fr] == 1]
        out[int(t)] = (idx, int(cnt[trunk].sum()) if len(trunk) else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--truth-csv", required=True)
    ap.add_argument("--nsp-threshold", type=int, default=80)
    ap.add_argument("--own-frac", type=float, default=0.5,
                    help="photon-fraction of its slice above which it's its OWN slice")
    ap.add_argument("--out", default="workdir_scale/photon_slice.csv")
    args = ap.parse_args()

    truth = {}
    with open(args.truth_csv) as fh:
        for r in csv.DictReader(fh):
            truth[(int(r["run"]), int(r["subrun"]), int(r["event"]))] = int(r["is_1g0X"])

    # build merged-file basename -> path index once
    merged = {}
    for p in glob.glob(os.path.join(args.merged_dir, "**", "merged_*entry*.h5"), recursive=True):
        merged[os.path.basename(p)] = p

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "**", "stage3pred_*.h5"), recursive=True))
    rows = []
    for pf in preds:
        base = os.path.basename(pf)[len("stage3pred_"):]
        mp = merged.get(base)
        if mp is None:
            continue
        with h5py.File(pf, "r") as ph5:
            a = ph5.attrs
            run = int(a.get("meta_run", -1)); sub = int(a.get("meta_subrun", -1))
            evt = int(a.get("meta_event", -1))
            if "post" not in ph5 or "queries" not in ph5:
                continue
            post_coord = ph5["post"]["coord"][:]
            post_q = ph5["post"]["pred_query"][:]
            qcls = ph5["queries"]["class_argmax"][:]
        # voxel hash: post voxel -> slice (pred_query). majority per voxel.
        vox2q = {}
        for k, q in zip(vkey(post_coord), post_q):
            vox2q.setdefault(k, []).append(int(q))
        vox2q = {k: Counter(v).most_common(1)[0][0] for k, v in vox2q.items()}
        slice_total = Counter(int(q) for q in post_q)   # points per slice

        with h5py.File(mp, "r") as mh5:
            e = mh5["entry_0"]
            frags = photon_frag_indices(e)
            pos = e["triplet_data"]["pos"][:]
            mt = e["mc_particle_tree"]
            tid2E = {int(t): float(en) for t, en in zip(mt["trackid"][:], mt["energy_mev"][:])}

        is1g0X = truth.get((run, sub, evt), 0)
        for t, (idx, nsp) in frags.items():
            if nsp < args.nsp_threshold:
                continue
            qs = [vox2q[k] for k in vkey(pos[idx]) if k in vox2q]
            n_in_slicer = len(qs)
            if n_in_slicer == 0:
                cat = "deghosted/lost"; sl = -1; cls = -1; frac = 0.0
            else:
                sl, nph = Counter(qs).most_common(1)[0]     # dominant slice, photon pts in it
                cls = int(qcls[sl])
                frac = nph / float(slice_total[sl]) if slice_total[sl] else 0.0
                if cls == NU_SLICE_CLASS:
                    cat = "nu_slice"
                elif frac >= args.own_frac:
                    cat = "own_cosmic_slice"
                else:
                    cat = "merged_cosmic"
            rows.append((run, sub, evt, t, tid2E.get(t, -1.0), nsp, is1g0X,
                         cls, round(frac, 3), n_in_slicer, len(idx), cat))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("run,subrun,event,trackid,trueE_mev,nSP_trunk,is_1g0X,"
                 "slice_class,photon_frac_of_slice,n_pts_in_slicer,n_photon_pts,category\n")
        for r in rows:
            fh.write("%d,%d,%d,%d,%.3f,%d,%d,%d,%.3f,%d,%d,%s\n" % r)

    # ---- summary ----
    def tally(sub):
        c = Counter(r[11] for r in sub)
        n = len(sub)
        return n, c
    print("=" * 64)
    print("photon slice-level outcome  (own-frac>=%.2f, nu_slice=class0)" % args.own_frac)
    for label, sub in [("ALL visible photons", rows),
                       ("photons in 1g0X events", [r for r in rows if r[6] == 1])]:
        n, c = tally(sub)
        print("-" * 64)
        print("%s : %d" % (label, n))
        for cat in ["nu_slice", "own_cosmic_slice", "merged_cosmic", "deghosted/lost"]:
            print("  %-18s %5d  (%.1f%%)" % (cat, c.get(cat, 0), 100.0 * c.get(cat, 0) / max(n, 1)))
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
