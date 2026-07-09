"""Apply the LArPID CNN to a nu_reco shard, writing a nu_reco_larpid copy
with per-particle score datasets appended.

Per reco particle: prong pixels from its keypoint2 instance (part_inst_idx ->
particle/{i}/point_idx -> triplet_imgpix_index); context pixels from the
UNION of the event's accepted slices (this stream's slice + the other
stream's slice when present) — the slice-union acceptance that replaces the
legacy thrumu cosmic veto. Crop center: track -> far end of the fitted
polyline; shower -> the reco start point. Particles with < 10 prong pixels
in any plane are left unclassified (legacy gate).

    PYTHONPATH=./ python3 lartpc/larformer_reco/larpid/apply_larpid.py \
        --nu-reco-shard .../nu_reco_shard0000000.h5 \
        --kp2-list <same list the nu_reco run used> \
        --merged-sp-list <merged_sp list> \
        --out .../nu_reco_larpid_shard0000000.h5 \
        [--checkpoint ... | --sample-tag mcc9_..._run3b_...] [--device cuda]

Appended per event group (row-aligned with the part_* table):
  larpid_classified (P,) i32   1 = scored, 0 = failed the pixel gate
  larpid_scores (P,5) f32      PID log-softmax [e, gamma, mu, pi, p]
  larpid_completeness/purity (P,) f32
  larpid_process_scores (P,3) f32   [primary, from-neutral, from-charged]
  larpid_pid (P,) i32          argmax PDG (11/22/13/211/2212; -1 unclassified)
  larpid_process (P,) i32      argmax process code (-1 unclassified)
File attrs: larpid_checkpoint, larpid_pixel_threshold.
"""
import os
import re
import sys
import argparse
import glob

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from lartpc.larformer_reco.larpid.crop import (  # noqa: E402
    EventImages, crop_bounds, build_input, PIX_THRESHOLD, MIN_PRONG_PIXELS)
from lartpc.larformer_reco.larpid.model import (  # noqa: E402
    LArPID, select_checkpoint)
from lartpc.larformer_reco.utils import read_list  # noqa: E402


def _attr_str(attrs, k):
    v = attrs.get(k, "")
    return v.decode() if isinstance(v, bytes) else v


def _counterpart_kp2(kp_path):
    """The other stream's keypoint2 file for the same event, if present
    (keypoint2_event{i}_0.h5 <-> keypoint2_event{i}_fm_0.h5)."""
    base = os.path.basename(kp_path)
    if "_fm_" in base:
        other = base.replace("_fm_", "_")
    else:
        other = re.sub(r"(event\d+)_(\d+\.h5)$", r"\1_fm_\2", base)
    p = os.path.join(os.path.dirname(kp_path), other)
    return p if (p != kp_path and os.path.exists(p)) else None


def _particle_polys(g):
    """Unpack part_poly_cm into per-particle polylines."""
    npoly = g["part_npoly"][()]
    poly = g["part_poly_cm"][()]
    return np.split(poly, np.cumsum(npoly)[:-1]) if len(npoly) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nu-reco-shard", required=True)
    ap.add_argument("--kp2-list", required=True,
                    help="the SAME keypoint2 list the nu_reco run used "
                         "(gidx = line index)")
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint path (overrides --sample-tag)")
    ap.add_argument("--sample-tag", default=None,
                    help="sample name for run-based checkpoint selection "
                         "('run3' -> alternate weights). Default: inferred "
                         "from the merged-sp list filename.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tag = args.sample_tag or os.path.basename(args.merged_sp_list)
    ckpt = args.checkpoint or select_checkpoint(tag)
    print(f">>> checkpoint: {ckpt} (tag: {tag})")
    model = LArPID(ckpt, device=args.device)

    kp_list = read_list(args.kp2_list)
    msp_map = {os.path.basename(p): p for p in read_list(args.merged_sp_list)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_ev = n_scored = n_gate = 0
    with h5py.File(args.nu_reco_shard, "r") as fin, \
         h5py.File(args.out, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["larpid_checkpoint"] = model.checkpoint
        fout.attrs["larpid_pixel_threshold"] = PIX_THRESHOLD

        for ev in fin:
            fin.copy(fin[ev], fout, ev)
            g = fout[ev]
            P = int(g.attrs["n_particles"])
            out = {
                "larpid_classified": np.zeros(P, np.int32),
                "larpid_scores": np.full((P, 5), np.nan, np.float32),
                "larpid_completeness": np.full(P, np.nan, np.float32),
                "larpid_purity": np.full(P, np.nan, np.float32),
                "larpid_process_scores": np.full((P, 3), np.nan, np.float32),
                "larpid_pid": np.full(P, -1, np.int32),
                "larpid_process": np.full(P, -1, np.int32),
            }
            n_ev += 1
            try:
                gidx = int(ev.split("_")[-1])
                kp_path = kp_list[gidx]
                src = _attr_str(g.attrs, "src_file")
                msp_path = msp_map.get(src)
                if msp_path is None or P == 0:
                    raise KeyError(f"no merged_sp for {src}")
                with h5py.File(msp_path, "r") as fmsp:
                    ev_img = EventImages(fmsp["entry_0"])

                # --- slice-union acceptance (this stream + counterpart) ------
                union = []
                for p in [kp_path, _counterpart_kp2(kp_path)]:
                    if p is None:
                        continue
                    with h5py.File(p, "r") as fkp:
                        union.append(ev_img.triplet_rows(
                            fkp["slice/coord_cm"][()]))
                union_rows = (np.unique(np.concatenate(union))
                              if union else np.zeros(0, np.int64))
                ctx_pix = [ev_img.pixels_for(union_rows, p) for p in range(3)]

                # --- per-particle crops --------------------------------------
                kind = g["part_kind"][()]
                inst = g["part_inst_idx"][()]
                start = g["part_start_cm"][()]
                polys = _particle_polys(g)
                images, rows_scored = [], []
                with h5py.File(kp_path, "r") as fkp:
                    slice_coords = fkp["slice/coord_cm"][()]
                    for i in range(P):
                        if inst[i] < 0:
                            continue
                        pidx = fkp[f"particle/{inst[i]}/point_idx"][()]
                        trip = ev_img.triplet_rows(slice_coords[pidx])
                        if trip.size == 0:
                            continue
                        center3d = (polys[i][-1] if kind[i] == 0
                                    and len(polys[i]) else start[i])
                        prong_pix = [ev_img.pixels_for(trip, p)
                                     for p in range(3)]
                        bounds = crop_bounds(
                            prong_pix, ev_img.center_rowcol(trip, center3d))
                        # context restricted to the same crop windows
                        img, n_prong = build_input(prong_pix, ctx_pix, bounds)
                        if min(n_prong) < MIN_PRONG_PIXELS:
                            n_gate += 1
                            continue
                        images.append(img)
                        rows_scored.append(i)
                if images:
                    res = model(np.stack(images))
                    sel = np.asarray(rows_scored)
                    out["larpid_classified"][sel] = 1
                    out["larpid_scores"][sel] = res["class_scores"]
                    out["larpid_completeness"][sel] = res["completeness"]
                    out["larpid_purity"][sel] = res["purity"]
                    out["larpid_process_scores"][sel] = res["process_scores"]
                    out["larpid_pid"][sel] = res["pid"]
                    out["larpid_process"][sel] = res["process"]
                    n_scored += len(sel)
            except Exception as ex:
                print(f"  [warn] {ev}: {type(ex).__name__}: {ex}")
            for k, v in out.items():
                g.create_dataset(k, data=v)
    print(f">>> {n_ev} events: {n_scored} particles scored, "
          f"{n_gate} failed the pixel gate -> {args.out}")


if __name__ == "__main__":
    main()
