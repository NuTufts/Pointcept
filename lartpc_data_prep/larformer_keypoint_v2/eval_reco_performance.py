"""First-pass performance eval: per-species reco efficiency vs true KE.

Two metrics vs the same denominator (true neutrino-origin particles, from
merged_sp/mc_particle_tree, origin==1):
  A. segmentation: a predicted keypoint2 instance captured >=70% of the particle's
     slice points.
  B. attachment+kinematics: a nu_reco particle with reconstructed 4-momentum is
     attached (present in the nu_reco_shard output).
Linkage is by trackid: keypoint2.gt_trackid == nu_reco.part_gt_trackid ==
mc_particle_tree.trackid.  See performance_eval_spec.md.

    python eval_reco_performance.py \
      --keypoint2-list KP.txt --merged-sp-list MSP.txt \
      --nu-reco-dir output/nu_reco_valdata_all --out eval_records.npz [--plots DIR]

Pure h5py+numpy (+matplotlib if --plots); no torch/ROOT.
"""
import os
import glob
import argparse

import numpy as np
import h5py

PID2SP = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
          211: "pi", -211: "pi", 2212: "p"}
SPECIES = ["e", "gamma", "mu", "pi", "p"]
SP2I = {s: i for i, s in enumerate(SPECIES)}
MASS = {"e": 0.511, "gamma": 0.0, "mu": 105.6584, "pi": 139.5704, "p": 938.2721}
CLASS_NAMES = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
COMPLETENESS = 0.70


def read_list(p):
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _attr_str(h5attrs, key):
    v = h5attrs.get(key, "")
    return v.decode() if isinstance(v, bytes) else v


def build_kp_index(kp_list):
    """src_file basename -> (gidx, kp_path). gidx = line index in the list."""
    idx = {}
    for gidx, kp in enumerate(kp_list):
        try:
            with h5py.File(kp, "r") as f:
                src = _attr_str(f.attrs, "src_file")
            if src:
                idx[os.path.basename(src)] = (gidx, kp)
        except Exception:
            continue
    return idx


def preload_nu_reco(nu_reco_dir):
    """gidx -> dict of part arrays, from all nu_reco_shard*.h5."""
    out = {}
    for shard in sorted(glob.glob(os.path.join(nu_reco_dir, "*.h5"))):
        try:
            with h5py.File(shard, "r") as f:
                for ev in f:
                    g = f[ev]
                    gidx = int(ev.split("_")[-1])
                    out[gidx] = dict(
                        gt=np.asarray(g["part_gt_trackid"][()], np.int64),
                        energy=np.asarray(g["part_energy"][()], np.float64),
                        cls=np.asarray(g["part_pred_class"][()], np.int64),
                        kind=np.asarray(g["part_kind"][()], np.int64))
        except Exception as e:
            print(f"  [warn] {shard}: {e}")
    return out


def completeness_by_trackid(kp_path):
    """{trackid: max completeness over instances, pred_class of best instance}."""
    best = {}
    with h5py.File(kp_path, "r") as f:
        n = int(f.attrs["n_particles"])
        for i in range(n):
            g = f[f"particle/{i}"]
            t = int(g.attrs["gt_trackid"])
            gp = g["gt_point_idx"][()]
            if t <= 0 or gp.size == 0:
                continue
            inter = np.intersect1d(g["point_idx"][()], gp, assume_unique=False).size
            c = inter / gp.size
            if t not in best or c > best[t][0]:
                best[t] = (c, int(g.attrs["cls"]))
    return best


def reco_ke(energy, cls_idx):
    name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "other"
    return energy - MASS.get(name, 0.0) if name in ("mu", "pi", "p") else energy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoint2-list", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--nu-reco-dir", required=True)
    ap.add_argument("--out", default="eval_records.npz")
    ap.add_argument("--plots", default=None, help="dir for PNGs (needs matplotlib)")
    ap.add_argument("--primaries-only", action="store_true",
                    help="denominator = primaries (parent is the nu) only")
    args = ap.parse_args()

    msp_map = {os.path.basename(p): p for p in read_list(args.merged_sp_list)}
    kp_list = read_list(args.keypoint2_list)
    print(f">>> {len(kp_list)} keypoint2, {len(msp_map)} merged_sp; indexing...",
          flush=True)
    kp_index = build_kp_index(kp_list)
    nu_reco = preload_nu_reco(args.nu_reco_dir)
    print(f">>> kp_index={len(kp_index)}  nu_reco events={len(nu_reco)}", flush=True)

    # per-particle records (one row per true nu-origin particle)
    rec = dict(species=[], true_ke=[], found_A=[], compl=[], found_B=[],
               reco_ke=[], reco_class=[], had_kp=[])
    n_ev = 0
    for msp_base, msp_path in msp_map.items():
        try:
            with h5py.File(msp_path, "r") as f:
                mt = f["entry_0/mc_particle_tree"]
                tid = mt["trackid"][()]; pid = mt["pid"][()]
                ke = mt["energy_mev"][()]; org = mt["origin"][()]
                par = mt["parent_trackid"][()] if args.primaries_only else None
        except Exception:
            continue
        # true nu-origin particles of reconstructable species
        truth = {}
        for i in range(len(tid)):
            if org[i] != 1 or int(pid[i]) not in PID2SP:
                continue
            if args.primaries_only and int(par[i]) not in (0, int(tid[i])):
                continue
            truth[int(tid[i])] = (PID2SP[int(pid[i])], float(ke[i]))
        if not truth:
            continue
        n_ev += 1
        hit = kp_index.get(msp_base)
        compl = completeness_by_trackid(hit[1]) if hit else {}
        nr = nu_reco.get(hit[0]) if hit else None
        for t, (sp, tke) in truth.items():
            c = compl.get(t, (0.0, -1))[0]
            fB = rk = rc = None
            if nr is not None:
                m = np.where(nr["gt"] == t)[0]
                if m.size:
                    j = m[np.argmax(nr["energy"][m])]      # largest-energy match
                    rk = reco_ke(float(nr["energy"][j]), int(nr["cls"][j]))
                    rc = int(nr["cls"][j])
            rec["species"].append(SP2I[sp])
            rec["true_ke"].append(tke)
            rec["found_A"].append(c >= COMPLETENESS)
            rec["compl"].append(c)
            rec["found_B"].append(rk is not None)
            rec["reco_ke"].append(rk if rk is not None else 0.0)
            rec["reco_class"].append(rc if rc is not None else -1)
            rec["had_kp"].append(hit is not None)
    for k in rec:
        rec[k] = np.asarray(rec[k])
    np.savez(args.out, species_names=np.array(SPECIES), **rec)
    print(f">>> {n_ev} events with nu-origin truth, {len(rec['species'])} "
          f"true particles -> {args.out}", flush=True)

    # ---- summary: efficiency in coarse KE bins ----
    bins = [0, 50, 100, 200, 400, 800, 1e9]
    print("\nspecies | KE bin[MeV] |   N  | eff_A(>=70%) | eff_B(attach)")
    for si, sp in enumerate(SPECIES):
        m = rec["species"] == si
        if m.sum() == 0:
            continue
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = m & (rec["true_ke"] >= lo) & (rec["true_ke"] < hi)
            N = int(b.sum())
            if N == 0:
                continue
            eA = rec["found_A"][b].mean(); eB = rec["found_B"][b].mean()
            print(f"{sp:>6} | {lo:5.0f}-{hi if hi < 1e8 else 9999:<5.0f} | "
                  f"{N:5d} | {eA:5.2f}       | {eB:5.2f}")

    if args.plots:
        _plots(rec, args.plots)


def _plots(rec, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    edges = np.linspace(0, 1000, 21)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for si, sp in enumerate(SPECIES):
        m = rec["species"] == si
        if m.sum() < 5:
            continue
        # efficiency vs KE
        fig, ax = plt.subplots(figsize=(5, 4))
        for key, lab in (("found_A", "A: seg >=70%"), ("found_B", "B: attached")):
            eff, err = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = m & (rec["true_ke"] >= lo) & (rec["true_ke"] < hi)
                N = b.sum()
                e = rec[key][b].mean() if N else np.nan
                eff.append(e)
                err.append(np.sqrt(e * (1 - e) / N) if N else 0)
            ax.errorbar(ctr, eff, yerr=err, marker="o", ms=3, label=lab)
        ax.set(xlabel="true KE [MeV]", ylabel="efficiency", ylim=(0, 1.05),
               title=f"{sp} reco efficiency")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{outdir}/eff_{sp}.png", dpi=110)
        plt.close(fig)
        # 2D reco vs true KE (missed -> 0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist2d(rec["true_ke"][m], rec["reco_ke"][m],
                  bins=[np.linspace(0, 1000, 40)] * 2, cmin=1)
        ax.plot([0, 1000], [0, 1000], "r--", lw=1)
        ax.set(xlabel="true KE [MeV]", ylabel="reco KE [MeV]",
               title=f"{sp}: reco vs true KE (missed=0)")
        fig.tight_layout(); fig.savefig(f"{outdir}/recovstrue_{sp}.png", dpi=110)
        plt.close(fig)
    print(f">>> plots -> {outdir}")


if __name__ == "__main__":
    main()
