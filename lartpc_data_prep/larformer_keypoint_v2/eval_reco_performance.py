"""First-pass performance eval: per-species reco efficiency vs true KE.

Three metrics vs the same denominator (true neutrino-origin particles, from
merged_sp/mc_particle_tree, origin==1):
  A. segmentation: a predicted keypoint2 instance captured >=70% of the particle's
     slice points.
  B. attachment+kinematics: a nu_reco particle with reconstructed 4-momentum is
     attached (present in the nu_reco_shard output).
  C. slice coverage (upstream): the nu slice (slice/coord_cm) captures >=50% of the
     particle's visible ionization -- CHARGE-based, = (de-double-counted pixel
     charge of its in-slice spacepoints) / (that of all its true spacepoints), using
     the reco's shower-charge sum (calo.dedup_charge, Y else mean(U,V)). Charge, not
     spacepoint count, so it isn't fooled by the GT labeller being over-liberal on
     the low-charge ionization edges/tails. Isolates upstream slice/deghost losses
     from the reco -- if C is high but A/B are low the problem is in the reco; if C
     itself is low the particle never made it into the slice. (slice_cov_count keeps
     the old count-based coverage for comparison.)
Linkage is by trackid: keypoint2.gt_trackid == nu_reco.part_gt_trackid ==
mc_particle_tree.trackid == triplet_data.trackid.  See performance_eval_spec.md.

    python eval_reco_performance.py \
      --keypoint2-list KP.txt --merged-sp-list MSP.txt \
      --nu-reco-dir output/nu_reco_valdata_all --out eval_records.npz [--plots DIR]

Pure h5py+numpy+scipy (+matplotlib if --plots); no torch/ROOT. Reuses the reco's
de-double-counted calorimetric charge (trajfit_dev/calo.py) for metric C.
"""
import os
import sys
import glob
import argparse

import numpy as np
import h5py

# reuse the reco's de-double-counted calorimetric charge for metric C (the same
# `comb` = Y-plane-else-mean(U,V) charge the shower energy reco integrates).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trajfit_dev"))
from calo import dedup_charge  # noqa: E402

PID2SP = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
          211: "pi", -211: "pi", 2212: "p"}
SPECIES = ["e", "gamma", "mu", "pi", "p"]
SP2I = {s: i for i, s in enumerate(SPECIES)}
MASS = {"e": 0.511, "gamma": 0.0, "mu": 105.6584, "pi": 139.5704, "p": 938.2721}
CLASS_NAMES = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
COMPLETENESS = 0.70
SLICE_COVERAGE = 0.50


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
    """({trackid: (max completeness, pred_class of best instance)}, slice_coord)."""
    best = {}
    with h5py.File(kp_path, "r") as f:
        slice_coord = f["slice/coord_cm"][()].astype(np.float32)
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
    return best, slice_coord


def _charge_sum(pixval, tick, uw, vw, yw, sel):
    """De-double-counted unique-pixel charge (comb: Y else mean(U,V)) over `sel`.

    calo.dedup_charge splits each wire pixel's ADC among the spacepoints sharing
    it, so summing the per-point charge over a set counts every pixel it touches
    exactly once -- the same quantity the shower energy reco integrates.
    """
    if not np.any(sel):
        return 0.0
    _, q_comb = dedup_charge(pixval[sel], tick[sel], uw[sel], vw[sel], yw[sel])
    return float(q_comb.sum())


def slice_coverage(entry, slice_coord, truth_ids):
    """Per-trackid charge- and count-based nu-slice coverage from triplet_data.

    Returns four {trackid: value} dicts:
      n_true : # triplet_data spacepoints truth-matched to the particle (count)
      q_true : de-double-counted unique-pixel CHARGE of those points (denominator)
      q_slice: charge of the subset that fall inside the nu slice (numerator)
      n_slice: count of those in-slice points (for the count-based comparison)

    The headline coverage (metric C) is CHARGE-based, q_slice/q_true: the fraction
    of the particle's visible ionization the slice kept. This is robust to the GT
    labeller being over-liberal on the low-charge ionization edges/tails (lots of
    edge spacepoints, little charge) that count-based coverage over-penalises.
    Charge is the reco's `comb` (Y plane else mean(U,V)); denominator = "pass in
    the GT spacepoints", numerator = "pass in the reco/slice spacepoints", each
    de-double-counted over its own set. Slice membership is an exact position
    match triplet_data->slice (bit-identical subset, dist 0).
    """
    td = entry["triplet_data"]
    td_tid = np.asarray(td["trackid"][()], np.int64)
    sel = np.isin(td_tid, np.asarray(list(truth_ids), np.int64))
    n_true, q_true, q_slice, n_slice = {}, {}, {}, {}
    if not sel.any():
        return n_true, q_true, q_slice, n_slice
    tid_sel = td_tid[sel]
    pixval = np.asarray(td["pixval"][()])[sel]
    tick = np.asarray(td["tick"][()])[sel]
    uw = np.asarray(td["uwire"][()])[sel]
    vw = np.asarray(td["vwire"][()])[sel]
    yw = np.asarray(td["ywire"][()])[sel]
    inm = np.zeros(len(tid_sel), bool)
    if slice_coord is not None and len(slice_coord):
        from scipy.spatial import cKDTree
        pos_sel = np.asarray(td["pos"][()], np.float32)[sel]
        d, _ = cKDTree(slice_coord).query(pos_sel, k=1)
        inm = d < 0.05   # cm; slice points are bit-identical members of triplet_data
    for t in np.unique(tid_sel):
        m = tid_sel == t
        ms = m & inm
        ti = int(t)
        n_true[ti] = int(m.sum())
        n_slice[ti] = int(ms.sum())
        q_true[ti] = _charge_sum(pixval, tick, uw, vw, yw, m)
        q_slice[ti] = _charge_sum(pixval, tick, uw, vw, yw, ms)
    return n_true, q_true, q_slice, n_slice


def reco_ke(energy, cls_idx):
    name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "other"
    return energy - MASS.get(name, 0.0) if name in ("mu", "pi", "p") else energy


PI0 = 111


def is_primary(t, pid_by_tid, par_by_tid):
    """True if particle `t` is a ν-vertex primary.

    A ν primary has parent_trackid 0 (or itself). Photons from a primary π0
    decay are also treated as primaries: the π0 decays effectively at the ν
    vertex, so its two γ appear to originate there. π0 itself is not
    reconstructable (not in PID2SP) so it never enters the denominator.
    """
    p = int(par_by_tid.get(t, 0))
    if p in (0, t):
        return True
    # γ from a π0 whose own parent is the ν (primary π0)
    if int(pid_by_tid.get(t, 0)) == 22 and int(pid_by_tid.get(p, 0)) == PI0:
        pp = int(par_by_tid.get(p, 0))
        return pp in (0, p)
    return False


RECORD_KEYS = ["species", "true_ke", "found_A", "compl", "found_B",
               "reco_ke", "reco_class", "had_kp", "found_C", "slice_cov",
               "slice_cov_count", "q_true", "n_true_sp", "has_instance"]

# per-particle failure-stage diagnosis (mirrors the visualizer's why_unattached):
# where in the pipeline each true particle fell out. First matching wins.
STAGE_NAMES = ["ok", "misID", "seg!att", "noInst", "missSlice"]
STAGE_DESC = {
    "ok":        "attached + correct class",
    "misID":     "attached but wrong predicted class",
    "seg!att":   "segmenter made an instance but reco did not attach it",
    "noInst":    "in slice (charge) but segmenter made no instance",
    "missSlice": "little/no charge in the slice (upstream loss)",
}
STAGE_LOWCOV = 0.10   # charge slice-coverage below which we call it "missed"


def stage_codes(rec, lowcov=STAGE_LOWCOV):
    """int stage per particle: 0 ok, 1 misID, 2 seg!att, 3 noInst, 4 missSlice."""
    n = len(rec["species"])
    fb = np.asarray(rec["found_B"]).astype(bool)
    hi = np.asarray(rec["has_instance"]).astype(bool)
    cov = np.asarray(rec["slice_cov"], float)
    rc = np.asarray(rec["reco_class"], int)
    sp = np.asarray(rec["species"], int)
    out = np.empty(n, int)
    for i in range(n):
        if fb[i]:
            cn = CLASS_NAMES[rc[i]] if 0 <= rc[i] < len(CLASS_NAMES) else "?"
            out[i] = 0 if cn == SPECIES[sp[i]] else 1
        elif hi[i]:
            out[i] = 2
        elif cov[i] >= lowcov:
            out[i] = 3
        else:
            out[i] = 4
    return out


def print_summary(rec):
    """Per-species efficiency + failure-stage breakdown in coarse KE bins."""
    bins = [0, 50, 100, 200, 400, 800, 1e9]
    print("\nspecies | KE bin[MeV] |   N  | eff_A(seg70) | eff_B(attach) | "
          "eff_C(chgQ50)")
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
            eC = rec["found_C"][b].mean()
            print(f"{sp:>6} | {lo:5.0f}-{hi if hi < 1e8 else 9999:<5.0f} | "
                  f"{N:5d} | {eA:5.2f}        | {eB:5.2f}         | {eC:5.2f}")
    if "has_instance" in rec:
        print_stage_breakdown(rec, bins)


def print_stage_breakdown(rec, bins):
    """Fraction of true particles in each reco failure stage, per species/KE bin."""
    codes = stage_codes(rec)
    print("\n== reco failure-stage breakdown (fraction of true particles) ==")
    print("   " + "  ".join(f"{s}={STAGE_DESC[s]}" for s in STAGE_NAMES))
    print(f"{'species':>6} | {'KE bin[MeV]':>11} | {'N':>5} | " +
          " | ".join(f"{s:>9}" for s in STAGE_NAMES))
    for si, sp in enumerate(SPECIES):
        m = rec["species"] == si
        if m.sum() == 0:
            continue
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = m & (rec["true_ke"] >= lo) & (rec["true_ke"] < hi)
            N = int(b.sum())
            if N == 0:
                continue
            fr = [f"{(codes[b] == k).mean():9.2f}" for k in range(len(STAGE_NAMES))]
            print(f"{sp:>6} | {lo:5.0f}-{hi if hi < 1e8 else 9999:<5.0f} | "
                  f"{N:5d} | " + " | ".join(fr))


def merge_shards(glob_pat, out, plots):
    """Concatenate per-shard record npz files -> one table + summary + plots."""
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"no shard npz matched {glob_pat!r}")
    parts = {k: [] for k in RECORD_KEYS}
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            for k in RECORD_KEYS:
                parts[k].append(z[k])
    rec = {k: np.concatenate(parts[k]) if parts[k] else np.array([])
           for k in RECORD_KEYS}
    np.savez(out, species_names=np.array(SPECIES), **rec)
    print(f">>> merged {len(paths)} shards -> {len(rec['species'])} true "
          f"particles -> {out}", flush=True)
    print_summary(rec)
    if plots:
        _plots(rec, plots)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoint2-list")
    ap.add_argument("--merged-sp-list")
    ap.add_argument("--nu-reco-dir")
    ap.add_argument("--out", default="eval_records.npz")
    ap.add_argument("--plots", default=None, help="dir for PNGs (needs matplotlib)")
    ap.add_argument("--primaries-only", action="store_true",
                    help="denominator = primaries (parent is the nu) only")
    ap.add_argument("--start", type=int, default=0,
                    help="shard: first merged_sp index (sorted by basename)")
    ap.add_argument("--n", type=int, default=None,
                    help="shard: number of merged_sp files to process from --start")
    ap.add_argument("--merge", metavar="GLOB",
                    help="merge mode: concatenate matching shard npz -> --out "
                         "(+summary/plots); ignores the list/dir args")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.merge, args.out, args.plots)
        return

    for req in ("keypoint2_list", "merged_sp_list", "nu_reco_dir"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --merge)")

    # deterministic order so --start/--n shards are contiguous & non-overlapping
    msp_items = sorted({os.path.basename(p): p
                        for p in read_list(args.merged_sp_list)}.items())
    ntot = len(msp_items)
    lo = args.start
    hi = ntot if args.n is None else min(ntot, args.start + args.n)
    msp_items = msp_items[lo:hi]
    kp_list = read_list(args.keypoint2_list)
    print(f">>> {len(kp_list)} keypoint2, {ntot} merged_sp; "
          f"this shard: [{lo}:{hi}] ({len(msp_items)}); indexing...", flush=True)
    kp_index = build_kp_index(kp_list)
    nu_reco = preload_nu_reco(args.nu_reco_dir)
    print(f">>> kp_index={len(kp_index)}  nu_reco events={len(nu_reco)}", flush=True)

    # per-particle records (one row per true nu-origin particle)
    rec = {k: [] for k in RECORD_KEYS}
    n_ev = 0
    for msp_base, msp_path in msp_items:
        try:
            fmsp = h5py.File(msp_path, "r")
        except Exception:
            continue
        try:
            entry = fmsp["entry_0"]
            mt = entry["mc_particle_tree"]
            tid = mt["trackid"][()]; pid = mt["pid"][()]
            ke = mt["energy_mev"][()]; org = mt["origin"][()]
            par = mt["parent_trackid"][()] if args.primaries_only else None
        except Exception:
            fmsp.close()
            continue
        # maps for the primaries cut (need parent pid to catch π0-decay γ)
        if args.primaries_only:
            pid_by_tid = {int(tid[i]): int(pid[i]) for i in range(len(tid))}
            par_by_tid = {int(tid[i]): int(par[i]) for i in range(len(tid))}
        # true nu-origin particles of reconstructable species
        truth = {}
        for i in range(len(tid)):
            if org[i] != 1 or int(pid[i]) not in PID2SP:
                continue
            if args.primaries_only and not is_primary(int(tid[i]), pid_by_tid,
                                                       par_by_tid):
                continue
            truth[int(tid[i])] = (PID2SP[int(pid[i])], float(ke[i]))
        if not truth:
            fmsp.close()
            continue
        n_ev += 1
        hit = kp_index.get(msp_base)
        compl, slice_coord = completeness_by_trackid(hit[1]) if hit else ({}, None)
        nr = nu_reco.get(hit[0]) if hit else None
        # metric C: charge-based slice coverage of each true particle
        try:
            n_true_sp, q_true, q_slice, n_in_slice = slice_coverage(
                entry, slice_coord, truth.keys())
        except Exception as ex:
            print(f"  [warn] slice_coverage {msp_base}: {ex}")
            n_true_sp, q_true, q_slice, n_in_slice = {}, {}, {}, {}
        fmsp.close()
        for t, (sp, tke) in truth.items():
            c = compl.get(t, (0.0, -1))[0]
            fB = rk = rc = None
            if nr is not None:
                m = np.where(nr["gt"] == t)[0]
                if m.size:
                    j = m[np.argmax(nr["energy"][m])]      # largest-energy match
                    rk = reco_ke(float(nr["energy"][j]), int(nr["cls"][j]))
                    rc = int(nr["cls"][j])
            ntsp = int(n_true_sp.get(t, 0))
            qt = q_true.get(t, 0.0)
            cov = q_slice.get(t, 0.0) / qt if qt > 0 else 0.0    # charge coverage
            cov_count = n_in_slice.get(t, 0) / ntsp if ntsp > 0 else 0.0
            rec["species"].append(SP2I[sp])
            rec["true_ke"].append(tke)
            rec["found_A"].append(c >= COMPLETENESS)
            rec["compl"].append(c)
            rec["found_B"].append(rk is not None)
            rec["reco_ke"].append(rk if rk is not None else 0.0)
            rec["reco_class"].append(rc if rc is not None else -1)
            rec["had_kp"].append(hit is not None)
            rec["found_C"].append(cov >= SLICE_COVERAGE)
            rec["slice_cov"].append(cov)
            rec["slice_cov_count"].append(cov_count)
            rec["q_true"].append(qt)
            rec["n_true_sp"].append(ntsp)
            rec["has_instance"].append(t in compl)   # segmenter made an instance
    for k in rec:
        rec[k] = np.asarray(rec[k])
    np.savez(args.out, species_names=np.array(SPECIES), **rec)
    print(f">>> {n_ev} events with nu-origin truth, {len(rec['species'])} "
          f"true particles -> {args.out}", flush=True)

    print_summary(rec)

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
        for key, lab in (("found_C", "C: charge slice >=50%"),
                         ("found_A", "A: seg >=70%"),
                         ("found_B", "B: attached")):
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
        # slice-coverage distribution (metric C, upstream diagnostic)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(rec["slice_cov"][m], bins=np.linspace(0, 1, 21), color="C0")
        ax.axvline(SLICE_COVERAGE, color="r", ls="--", lw=1,
                   label=f"cut={SLICE_COVERAGE:.2f}")
        ax.set(xlabel="charge slice coverage of true ionization",
               ylabel="particles", title=f"{sp}: nu-slice charge coverage")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{outdir}/slicecov_{sp}.png", dpi=110)
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
    if "has_instance" in rec:
        _stage_plots(rec, outdir)
    print(f">>> plots -> {outdir}")


def _stage_plots(rec, outdir):
    """Stacked-bar of the reco failure-stage mix vs KE, per species -- shows which
    stage dominates the eff_B gap (e.g. electrons: mis-ID vs no-instance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = stage_codes(rec)
    bins = [0, 50, 100, 200, 400, 800, 1e9]
    labels = [f"{lo:.0f}-{hi if hi < 1e8 else 9999:.0f}"
              for lo, hi in zip(bins[:-1], bins[1:])]
    # ok green, misID orange, seg!att red, noInst blue, missSlice gray
    colors = ["#3ca03c", "#e08000", "#d03030", "#5060d0", "#909090"]
    for si, sp in enumerate(SPECIES):
        m = rec["species"] == si
        if m.sum() < 5:
            continue
        fracs = np.zeros((len(STAGE_NAMES), len(labels)))
        Ns = []
        for j, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            b = m & (rec["true_ke"] >= lo) & (rec["true_ke"] < hi)
            N = int(b.sum()); Ns.append(N)
            if N:
                for k in range(len(STAGE_NAMES)):
                    fracs[k, j] = (codes[b] == k).mean()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bottoms = np.zeros(len(labels))
        for k in range(len(STAGE_NAMES)):
            ax.bar(labels, fracs[k], bottom=bottoms, color=colors[k],
                   label=STAGE_NAMES[k], width=0.8)
            bottoms += fracs[k]
        for x, N in enumerate(Ns):
            ax.text(x, 1.01, f"N={N}", ha="center", va="bottom", fontsize=6)
        ax.set(ylabel="fraction of true particles", ylim=(0, 1.12),
               title=f"{sp}: reco failure-stage mix vs true KE")
        ax.legend(fontsize=7, ncol=5, loc="lower center",
                  bbox_to_anchor=(0.5, -0.32))
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
        ax.set_xlabel("true KE [MeV]")
        fig.tight_layout(); fig.savefig(f"{outdir}/stage_{sp}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
