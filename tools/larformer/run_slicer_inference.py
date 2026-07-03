"""
Inference + per-event diagnostics dump for the LArFormer cascaded slicer.

Thin CLI around the slicer inference helpers in
`pointcept/models/LArFormer/inference.py`. For each input merged_h5
event, runs the trained `CascadedSlicer` in eval mode and writes a
self-contained HDF5 file with everything needed to visualize and
analyze where the model's predictions go wrong:

  - pre-filter (raw input) per-SP arrays: coord, hasmatch, slice_id, lm_score,
    pixval, ssnet_label, deghoster's P(real), kept-by-deghoster mask
  - post-filter (slicer input) per-SP arrays + the slicer's per-SP
    predicted slice assignment (argmax over matched queries)
  - per-query info: class logits, argmax class, matched GT idx
  - per-GT-instance info: which query matched, per-pair mask IoU,
    per-pair class correctness, primary_trackid (= GT slice id)
  - event identity + summary stats (tau, keep_frac, n_matched, etc.)

The per-SP predicted slice assignment uses the standard Mask2Former inference
rule: for each spacepoint, pick the query whose mask logit is highest
*among queries whose argmax class is not no_object*. That query's matched
GT instance (if any) gives the predicted slice ID. SPs whose best query is
unmatched get pred_slice_id = -1 (= "unassigned").

Usage:
    ./run_in_container.sh python tools/larformer/run_slicer_inference.py \\
        --config configs/lartpc/larformer/stage2_slicer/archive/larformer-slicer-v1-cascaded-loradeghost.py \\
        --weights exp/larformer_slicer_v1_cascaded_loradeghost/model/model_last.pth \\
        --input-list devdata_mergedh5_pi0filter_10files.txt \\
        --output-dir exp/larformer_slicer_v1_cascaded_loradeghost/inference \\
        --max-events 10

Output file naming: `slicerpred_<input_basename_without_ext>.h5`, one per
input event. Self-contained — no need to keep the source merged_h5 around
to visualize the predictions.
"""

import argparse
import os
import sys

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Import the shared helpers from the LArFormer model package — same
# code path the full-cascade Stage-3 inference uses, so a slicerpred
# file from this CLI is byte-identical to the slicer portion of a
# stage3pred file from `run_larformer_stage3_inference.py`.
from pointcept.models.LArFormer.inference import (   # noqa: E402
    slicer_predict_event,
    write_event_h5,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", required=True,
                    help="Path to the cascaded slicer config")
    ap.add_argument("--weights", required=True,
                    help="Path to model_*.pth checkpoint")
    ap.add_argument("--input-list", required=True,
                    help="Text file: one merged_h5 path per line")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write per-event prediction HDF5 files")
    ap.add_argument("--max-events", type=int, default=None,
                    help="If set, stop after this many events")
    ap.add_argument("--split", default="val",
                    choices=("train", "val", "test"),
                    help="Which dataset split's kwargs to copy from the "
                         "config (only the data_list_file is overridden)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models  # noqa: F401 — registers MODELS

    cfg = Config.fromfile(args.config)

    # Dataset: copy the split's kwargs, override data_list_file.
    ds_cfg = dict(cfg.data[args.split])
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None      # no cap during inference — we want
                                          # the full event so the predictions
                                          # are useful for downstream analysis
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f"[infer] Dataset: {len(dataset)} events; will process {n_events}.")

    print(f"[infer] Building model from {args.config}")
    model = build_model(cfg.model).to(args.device).eval()

    print(f"[infer] Loading weights from {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded; missing={len(missing)}  unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    # The cascade's slicer.loss_fn has the no_object_class_id we need for
    # the per-SP slice-assignment rule. Reach through the wrapper.
    inner_model = getattr(model, "module", model)
    if hasattr(inner_model, "slicer"):
        slicer = inner_model.slicer
    else:
        slicer = inner_model     # plain LArFormer (no cascade)
    no_object_class_id = int(slicer.loss_fn.no_object_class_id)
    print(f"[infer] no_object_class_id = {no_object_class_id}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_dropped = 0
    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        batched = larformer_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)

        event_data = slicer_predict_event(
            model, sample, batched, no_object_class_id,
        )
        if event_data.get("event_dropped"):
            n_dropped += 1

        in_name = sample.get("name", f"event{ev_idx:06d}.h5")
        stem = os.path.splitext(in_name)[0]
        out_path = os.path.join(args.output_dir, f"slicerpred_{stem}.h5")
        write_event_h5(out_path, event_data)

        keep_frac = event_data.get("meta/keep_frac", float("nan"))
        n_pre = event_data.get("meta/n_sp_pre", 0)
        n_post = event_data.get("meta/n_sp_post", 0)
        n_gt = event_data.get("meta/n_gt_instances", 0)
        n_matched = event_data.get("meta/n_matched", 0)
        pair_iou = event_data.get("gt/pair_iou", np.zeros(0))
        valid_iou = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        mean_iou = float(np.mean(valid_iou)) if valid_iou.size > 0 else float("nan")
        print(
            f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  "
            f"n_sp {n_pre} → {n_post}  keep_frac={keep_frac:.3f}  "
            f"matched={n_matched}/{n_gt}  mean_pair_IoU={mean_iou:.3f}"
        )

    print(f"\n[infer] Done. Wrote {n_events} files to {args.output_dir} "
          f"({n_dropped} events were dropped by the cascade's empty-event guard).")


if __name__ == "__main__":
    main()
