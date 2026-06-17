"""End-to-end inference for the attempt-2 keypoint model (CascadedKeypoint).

Runs the frozen Stage-3 cascade (deghoster + slicer + particle segmenter) on raw
merged_h5 events, then the attempt-2 keypoint model on the nu slice, and writes
per-event keypoints (per-particle start/end + the dense nu vertex) in DETECTOR
CM to an H5 per event.

Weights (all exist as separate files):
  --cascade-config   : larformer-particle-fullcascade-ptv3crosslevel.py (wires
                       deghoster + slicer + sonata via its weight knobs at build)
  --particle-weights : trained particle segmenter (epoch_6.pth) → particle_segmenter
  --keypoint-config  : larformer-keypoint2-particle-v1.py (the attempt-2 model)
  --keypoint-weights : trained keypoint model (model_best.pth) → keypoint_model
  --input-list       : text file of raw merged_h5 paths (e.g.
                       devdata_mergedh5_pi0filter_10files.txt)

Decode: per particle, the highest-prob START and END queries give start/end;
positions (recentered coord_norm) → cm via the per-event affine recovered from
(ps_coord, ps_coord_norm). The dense nu-vertex head is decoded as the
score-weighted centroid of spacepoints above --nu-thresh.
"""
import argparse
import os

import numpy as np
import torch

import pointcept.datasets  # noqa: F401
import pointcept.models.LArFormer  # noqa: F401  (registers CascadedKeypoint)
from pointcept.utils.config import Config
from pointcept.datasets import build_dataset, larformer_collate
from pointcept.models.builder import build_model
from pointcept.models.LArFormer.keypoint_eval import _recover_affine
from pointcept.models.LArFormer.keypoint2_particle import (
    KP_CLS_START, KP_CLS_END)
from tools.run_larformer_stage3_inference import (
    _load_weights_into, set_deterministic, reseed_per_event)


def _decode_event(ev, coord_cm, coord_norm, nu_thresh):
    """Per-event keypoints in cm. Returns dict of arrays."""
    scale, center = _recover_affine(coord_cm, coord_norm)
    if scale is None:
        scale = np.ones(3, np.float32)
        center = np.zeros(3, np.float32)

    def to_cm(p):
        return p * scale + center

    starts, ends, classes = [], [], []
    for p in ev.get("particle_kp", []):
        prob = p["class_logits"].softmax(-1).cpu().numpy()
        pos = p["pos"].cpu().numpy()
        si = int(prob[:, KP_CLS_START].argmax())
        starts.append(to_cm(pos[si]))
        classes.append(int(p.get("pred_class", -1)))
        # end only if some query prefers END over no_object
        ei = int(prob[:, KP_CLS_END].argmax())
        ends.append(to_cm(pos[ei]) if prob[ei, KP_CLS_END] > prob[ei, -1]
                    else np.full(3, np.nan, np.float32))

    nu = np.full(3, np.nan, np.float32)
    nh = ev.get("level_kp", {}).get("nu_vertex")
    if nh is not None:
        sc = nh["score"].cpu().numpy()
        cn = nh["coords"].cpu().numpy()
        sel = sc > nu_thresh
        nv_norm = ((cn[sel] * sc[sel, None]).sum(0) / sc[sel].sum()
                   if sel.any() else cn[int(sc.argmax())])
        nu = to_cm(nv_norm)

    return {
        "particle_start_cm": np.asarray(starts, np.float32).reshape(-1, 3),
        "particle_end_cm": np.asarray(ends, np.float32).reshape(-1, 3),
        "particle_class": np.asarray(classes, np.int64),
        "nu_vertex_cm": nu.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cascade-config", required=True)
    ap.add_argument("--particle-weights", required=True)
    ap.add_argument("--keypoint-config", required=True)
    ap.add_argument("--keypoint-weights", required=True)
    ap.add_argument("--input-list", required=True)
    ap.add_argument("--output-dir", default="kp2_cascade_out")
    ap.add_argument("--n-events", type=int, default=-1)
    ap.add_argument("--nu-thresh", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true",
                    help="Bit-exact inference on a fixed GPU/driver/lib stack "
                         "(TF32 off, deterministic algorithms, seeded, "
                         "CUBLAS_WORKSPACE_CONFIG). See "
                         "docs/LArFormer_Reproducibility.md. ~1.3-2x slower.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # MUST run before the model is built / any cuBLAS call (sets env + flags).
    if args.deterministic:
        set_deterministic(args.seed)
        print(">>> deterministic mode ON")

    casc = Config.fromfile(args.cascade_config)
    kp = Config.fromfile(args.keypoint_config)

    print(">>> building CascadedKeypoint ...")
    model = build_model(dict(
        type="CascadedKeypoint",
        cascade=dict(casc.model),
        keypoint_model=dict(kp.model),
        particle_source="predicted",
        no_object_class_id=int(kp.model.get("num_classes", 8)) - 1,
    ))
    _load_weights_into(model.cascade.particle_segmenter, args.particle_weights)
    _load_weights_into(model.keypoint_model, args.keypoint_weights)
    model = model.to(args.device).eval()

    ds_cfg = dict(casc.data.test)
    ds_cfg["data_root"] = "/"
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds = build_dataset(ds_cfg)
    n = len(ds) if args.n_events < 0 else min(args.n_events, len(ds))
    os.makedirs(args.output_dir, exist_ok=True)
    # The dataset SORTS its file list (get_data_list -> sorted), so ds[i] is the
    # i-th SORTED file, not the i-th input line — processing order is canonical
    # regardless of input order. Label outputs by the actual processed file.
    real_files = list(getattr(ds, "data_list", []))
    print(f">>> {n} events")

    import h5py
    for i in range(n):
        # Per-event order invariance (shared helper; see its docstring).
        if args.deterministic:
            reseed_per_event(args.seed)
        batch = larformer_collate([ds[i]])
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        preds = out.get("predictions", [])
        if not preds:
            print(f"  [{i}] no nu slice — skipped")
            continue
        offset = out["ps_offset"]
        coord = out["ps_coord"].cpu().numpy()
        coord_norm = out["ps_coord_norm"].cpu().numpy()
        for ei, ev in enumerate(preds):
            a = int(offset[ei - 1].item()) if ei > 0 else 0
            b = int(offset[ei].item())
            kp_cm = _decode_event(ev, coord[a:b], coord_norm[a:b], args.nu_thresh)
            outp = os.path.join(args.output_dir,
                                f"keypoint2_event{i:05d}_{ei}.h5")
            with h5py.File(outp, "w") as f:
                g = f.create_group("keypoints")
                for k, v in kp_cm.items():
                    g.create_dataset(k, data=v)
                # Event identity so a reorder test can match the same physical
                # event across runs (the loop index is order-dependent).
                f.attrs["src_file"] = os.path.basename(real_files[i]) \
                    if i < len(real_files) else ""
                for k in ("run", "subrun", "event"):
                    v = out.get(f"ps_{k}")
                    if v is not None:
                        f.attrs[k] = int(v[ei]) if hasattr(v, "__getitem__") \
                            else int(v)
            print(f"  [{i}.{ei}] n_particles={len(kp_cm['particle_class'])} "
                  f"nu_vertex_cm={kp_cm['nu_vertex_cm'].tolist()} -> {outp}")
    print("DONE")


if __name__ == "__main__":
    main()
