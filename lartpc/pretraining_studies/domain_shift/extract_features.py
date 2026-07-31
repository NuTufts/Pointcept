#!/usr/bin/env python3
"""
Batch frozen-feature extractor for the MC-data domain-shift study.

For every event in a file list, runs the teacher backbone of a Sonata
snapshot and writes per-event quantities to one .npz:

  pooled      [N_ev, C]     mean-pooled upcast features (C=1088 for the P05
                            base width) -- the event embedding used by
                            PAD/MMD/bootstrap (domain_metrics.py).
  proto_hist  [N_ev, P]     hard prototype-assignment histogram from the
                            teacher unmask head (P=4096) -- input to the
                            prototype-occupancy JSD.
  names, n_raw, n_kept, n_upcast, event level bookkeeping + a meta json.

Optionally (--points-per-event > 0) a fixed-seed per-point sample is saved
(float16) for point-level analyses / UMAP.

Tier definitions (see domain_shift_study_plan.md section 4):
  raw            no masking (Tier 0).
  cosmic         MC: drop origin==1 (nu-matched) points. Data: no-op. (Tier 1)
  cosmic-clean   cosmic + MC: keep hasmatch==1 (truth deghost); data: keep
                 lm score > --lm-threshold. (Tier 1, method-ASYMMETRIC clean)
  cosmic-lmclean cosmic + the SAME lm score > --lm-threshold cut on BOTH
                 domains (MC files carry larmatch inference as `lm_score`).
                 Method-symmetric clean: the difference to cosmic-clean
                 isolates cleaning-method asymmetry from genuine LArMatch
                 response mismodeling. Data side identical to cosmic-clean.

Measurement definition (fixed for the whole study -- do not vary):
  * The transform pipeline is the run config's val_transform with all
    SphereCrop-type transforms REMOVED: the embedding represents the WHOLE
    event, voxelized at the config grid_size. Events larger than
    --point-cap raw points are randomly subsampled (fixed per-event seed).
  * Tier masks are applied to the raw point arrays BEFORE voxelization.
  * Domain is decided per file: 'extbnb' in path => data, else mc (the
    make_filelists.py category rule).

Run inside the pointcept container on a GPU node, e.g.:
  apptainer exec --nv -B /cluster:/cluster $CONTAINER python3 \
      lartpc/pretraining_studies/domain_shift/extract_features.py \
      --config configs/lartpc/p05/pretrain-sonata-p5b1-mix-raw-detsym.py \
      --checkpoint sonata/p5b/P5B.1-mix_raw-s0/snapshot/snapshot_iter0128000_img6144000.pth \
      --data-list lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt \
      --tier cosmic --num-events 20 \
      --out lartpc/pretraining_studies/domain_shift/features/test_mc_cosmic.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from pointcept.utils.config import Config  # noqa: E402
from pointcept.models.builder import build_model  # noqa: E402
from pointcept.datasets.builder import DATASETS  # noqa: E402
from pointcept.datasets.transform import Compose  # noqa: E402

# Keys masked together with coord (mirrors the loader's own masking blocks).
MASKABLE_KEYS = ["coord", "strength", "color", "segment", "instance",
                 "origin", "hasmatch"]
# Raw per-point observables attached with --attach-observables. They are
# registered in index_valid_keys so GridSample keeps them point-aligned,
# then mapped input->upcast by nearest neighbor after the forward pass.
OBS_KEYS = ["obs_pixval", "obs_tick", "obs_lm", "obs_rawpos"]
# lm-score key candidates, in preference order (lm_score naming bug is
# documented in the experiment plan; EXTBNB v3 files carry larmatch_score).
LM_KEYS = ("larmatch_score", "lm_score")
MIN_POINTS = 64  # events with fewer post-mask raw points are skipped


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="run config the checkpoint was trained with "
                         "(defines the preprocessing incl. asinh vs log)")
    ap.add_argument("--checkpoint", required=True, help="snapshot .pth")
    ap.add_argument("--data-list", required=True, help="*_tufts.txt file list")
    ap.add_argument("--tier", choices=("raw", "cosmic", "cosmic-clean",
                                       "cosmic-lmclean"),
                    default="raw")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--num-events", type=int, default=0,
                    help="events to process (0 = whole list)")
    ap.add_argument("--start-event", type=int, default=0)
    ap.add_argument("--point-cap", type=int, default=120000,
                    help="max raw points fed to voxelization per event")
    ap.add_argument("--points-per-event", type=int, default=0,
                    help="per-point features to save per event (0 = off)")
    ap.add_argument("--lm-threshold", type=float, default=0.45,
                    help="data-side larmatch cut for tier cosmic-clean "
                         "(fixed, unlike the stochastic training-time cut)")
    ap.add_argument("--attach-observables", action="store_true",
                    help="save raw pos/pixval/lm/tick for each sampled "
                         "point (for DCTR closure studies); requires "
                         "--points-per-event > 0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    return ap.parse_args()


def is_data_file(path):
    return "extbnb" in path


def build_dataset(cfg, data_list):
    """LArTPCDataset with every loader-side filter OFF: tier masks are applied
    here in the extractor so event identity is never silently changed by the
    loader's too-few-points retry logic."""
    ds_cfg = cfg.data.val.copy()
    ds_cfg.update(dict(
        data_list_file=data_list,
        transform=[],
        test_mode=False,
        loop=1,
        true_points_only=False,
        filter_larmatch=False,
        drop_cosmics=False,
        # Force per-file truth probing regardless of the run config:
        # single-domain configs pin data_only True/False, which either
        # crashes on data files (False) or SILENTLY skips MC truth and
        # disables the origin/hasmatch tier masks (True).
        data_only="auto",
    ))
    ds_cfg.pop("ghost_keep_frac", None)
    return DATASETS.build(ds_cfg)


def build_transform(cfg, extra_keys=()):
    """val_transform minus SphereCrops, with bookkeeping keys collected and
    the scaled grid_size injected (required by PTv3 serialization after
    NormalizeCoord)."""
    scaled_grid_size = cfg.get("scaled_grid_size",
                               cfg.grid_size / cfg.coord_scale)
    tf = []
    for t in cfg.val_transform:
        ttype = t["type"]
        if "SphereCrop" in ttype:
            continue  # whole-event measurement definition
        if ttype == "Collect":
            t = dict(t)
            t["keys"] = ("coord", "grid_coord", "segment", "origin",
                         "hasmatch", "name", "grid_size") + tuple(extra_keys)
            tf.append(dict(type="Update",
                           keys_dict={"grid_size": scaled_grid_size}))
        tf.append(t)
    return Compose(tf)


def read_observables(h5_path, n_expected):
    """Raw per-point observables (full-file frame, pre-mask)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        td = f["/entry_0/triplet_data"]
        pixval = np.array(td["pixval"], dtype=np.float32)
        tick = np.array(td["tick"], dtype=np.int32)
    if pixval.shape[0] != n_expected:
        raise RuntimeError(
            f"{h5_path}: pixval len {pixval.shape[0]} != {n_expected}")
    return pixval, tick


def tier_mask(raw, tier, domain, lm_score, lm_threshold):
    """Boolean keep-mask over the raw (unfiltered) point arrays."""
    n = raw["coord"].shape[0]
    keep = np.ones(n, dtype=bool)
    if tier == "raw":
        return keep
    if domain == "mc":
        keep &= raw["origin"] != 1
        if tier == "cosmic-clean":
            keep &= raw["hasmatch"] == 1
        elif tier == "cosmic-lmclean":
            if lm_score is None:
                raise RuntimeError(
                    "cosmic-lmclean on MC requires an lm score")
            keep &= lm_score > lm_threshold
    elif tier in ("cosmic-clean", "cosmic-lmclean"):
        if lm_score is None:
            raise RuntimeError(f"{tier} on data requires an lm score")
        keep &= lm_score > lm_threshold
    return keep


def read_lm_score(h5_path, n_expected):
    import h5py
    with h5py.File(h5_path, "r") as f:
        td = f["/entry_0/triplet_data"]
        for k in LM_KEYS:
            if k in td:
                s = np.array(td[k], dtype=np.float32)
                if len(s) != n_expected:
                    raise RuntimeError(
                        f"{h5_path}: lm score len {len(s)} != {n_expected}")
                return s
    return None


def load_snapshot(cfg, ckpt_path, device):
    model = build_model(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_model = len(model.state_dict())
    n_loaded = n_model - len(missing)
    print(f"[load] {n_loaded}/{n_model} model keys matched "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    if n_loaded < 0.9 * n_model:
        raise RuntimeError(
            f"checkpoint/model mismatch: only {n_loaded}/{n_model} keys "
            f"loaded; first missing: {missing[:5]}")
    model = model.to(device).eval()
    images_seen = int(ckpt.get("images_seen", -1))
    return model, images_seen


@torch.no_grad()
def forward_event(model, batch, device):
    """Teacher backbone + upcast, plus teacher unmask-head prototype logits.
    Returns (upcast_feat [M,C] cpu f32, proto_assign [M] cpu int64)."""
    input_dict = {
        "coord": batch["coord"].to(device),
        "grid_coord": batch["grid_coord"].to(device),
        "feat": batch["feat"].to(device),
        "offset": torch.tensor([batch["coord"].shape[0]]).to(device),
        "grid_size": float(batch["grid_size"]),
    }
    point = model(input_dict, return_point=True)["point"]
    feat = point.feat
    head = model.teacher["unmask_head"] if "unmask_head" in model.teacher \
        else model.teacher["mask_head"]
    proto = head(feat).argmax(dim=-1)
    return feat.float().cpu(), proto.cpu(), point.coord.float().cpu()


def main():
    args = parse_args()
    t0 = time.time()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)

    dataset = build_dataset(cfg, args.data_list)
    transform = build_transform(
        cfg, extra_keys=OBS_KEYS if args.attach_observables else ())
    model, images_seen = load_snapshot(cfg, args.checkpoint, device)
    num_proto = cfg.model.head_num_prototypes

    n_total = len(dataset.data_list)
    end = n_total if args.num_events <= 0 \
        else min(args.start_event + args.num_events, n_total)
    idxs = range(args.start_event, end)
    print(f"[extract] {len(idxs)} events  tier={args.tier}  "
          f"list={os.path.basename(args.data_list)}")

    pooled, proto_hists, names, domains = [], [], [], []
    n_raw_l, n_kept_l, n_upcast_l = [], [], []
    pt_feats, pt_event, pt_origin, pt_segment, pt_proto, pt_coord = \
        [], [], [], [], [], []
    pt_pixval, pt_tick, pt_lm, pt_rawpos = [], [], [], []
    skipped = []

    for j, idx in enumerate(idxs):
        path = dataset.data_list[idx]
        domain = "data" if is_data_file(path) else "mc"
        try:
            raw = dataset.get_data(idx)
            n_raw = raw["coord"].shape[0]
            need_lm = (args.tier == "cosmic-lmclean"
                       or (args.tier == "cosmic-clean" and domain == "data"))
            lm = read_lm_score(path, n_raw) if need_lm else None
            keep = tier_mask(raw, args.tier, domain, lm, args.lm_threshold)

            rng = np.random.default_rng(args.seed + idx)
            if keep.sum() > args.point_cap:
                kept_idx = np.flatnonzero(keep)
                keep = np.zeros(n_raw, dtype=bool)
                keep[rng.choice(kept_idx, args.point_cap, replace=False)] = True
            n_kept = int(keep.sum())
            if n_kept < MIN_POINTS:
                skipped.append((raw["name"], f"{n_kept} pts after mask"))
                continue
            if args.attach_observables:
                pixval, tick = read_observables(path, n_raw)
                if lm is None:
                    lm = read_lm_score(path, n_raw)
                raw["obs_pixval"] = pixval
                raw["obs_tick"] = tick
                raw["obs_lm"] = lm if lm is not None else np.full(
                    n_raw, np.nan, dtype=np.float32)
                raw["obs_rawpos"] = raw["coord"].astype(np.float32).copy()
                raw["index_valid_keys"] = list(
                    raw.get("index_valid_keys", [])) + OBS_KEYS

            for k in MASKABLE_KEYS + (OBS_KEYS
                                      if args.attach_observables else []):
                if k in raw and hasattr(raw[k], "shape") \
                        and raw[k].shape[:1] == (n_raw,):
                    raw[k] = raw[k][keep]

            batch = transform(raw)
            feat, proto, coord = forward_event(model, batch, device)

            pooled.append(feat.mean(dim=0).numpy())
            proto_hists.append(
                np.bincount(proto.numpy(), minlength=num_proto
                            ).astype(np.int32))
            names.append(batch["name"])
            domains.append(domain)
            n_raw_l.append(n_raw)
            n_kept_l.append(n_kept)
            n_upcast_l.append(feat.shape[0])

            if args.points_per_event > 0:
                m = feat.shape[0]
                sel = rng.choice(m, min(args.points_per_event, m),
                                 replace=False)
                pt_feats.append(feat[sel].numpy().astype(np.float16))
                pt_event.append(np.full(len(sel), len(pooled) - 1,
                                        dtype=np.int32))
                pt_origin.append(
                    batch["origin"][sel].numpy().astype(np.int8))
                pt_segment.append(
                    batch["segment"][sel].numpy().astype(np.int16))
                pt_proto.append(proto[sel].numpy().astype(np.int32))
                pt_coord.append(coord[sel].numpy())
                if args.attach_observables:
                    # map upcast (stride-4) points back to input points by
                    # nearest neighbor in the model's normalized frame
                    from scipy.spatial import cKDTree
                    nn = cKDTree(batch["coord"].numpy()).query(
                        coord[sel].numpy())[1]
                    pt_pixval.append(
                        batch["obs_pixval"][nn].numpy().astype(np.float32))
                    pt_tick.append(
                        batch["obs_tick"][nn].numpy().astype(np.int32))
                    pt_lm.append(
                        batch["obs_lm"][nn].numpy().astype(np.float32))
                    pt_rawpos.append(
                        batch["obs_rawpos"][nn].numpy().astype(np.float32))
        except Exception as e:  # noqa: BLE001 -- record, never substitute
            skipped.append((os.path.basename(path), repr(e)))
            continue
        if (j + 1) % 50 == 0 or j == 0:
            dt = time.time() - t0
            print(f"  [{j + 1}/{len(idxs)}] {dt:.0f}s "
                  f"({dt / (j + 1):.2f}s/event)", flush=True)

    if not pooled:
        raise RuntimeError(f"no events extracted; skipped={skipped[:5]}")

    meta = dict(
        config=args.config, checkpoint=args.checkpoint,
        data_list=args.data_list, tier=args.tier,
        images_seen=images_seen, seed=args.seed,
        point_cap=args.point_cap, lm_threshold=args.lm_threshold,
        num_events_requested=len(idxs), num_events_kept=len(pooled),
        pooling="mean over upcast points",
        transform="val_transform minus SphereCrop, whole event",
    )
    out = dict(
        pooled=np.stack(pooled).astype(np.float32),
        proto_hist=np.stack(proto_hists),
        names=np.array(names), domains=np.array(domains),
        n_raw=np.array(n_raw_l, dtype=np.int64),
        n_kept=np.array(n_kept_l, dtype=np.int64),
        n_upcast=np.array(n_upcast_l, dtype=np.int64),
        skipped=np.array([f"{n}: {r}" for n, r in skipped]),
        meta=np.array(json.dumps(meta)),
    )
    if args.points_per_event > 0:
        out.update(
            point_feats=np.concatenate(pt_feats),
            point_event=np.concatenate(pt_event),
            point_origin=np.concatenate(pt_origin),
            point_segment=np.concatenate(pt_segment),
            point_proto=np.concatenate(pt_proto),
            point_coord=np.concatenate(pt_coord),
        )
        if args.attach_observables:
            out.update(
                point_pixval=np.concatenate(pt_pixval),
                point_tick=np.concatenate(pt_tick),
                point_lm=np.concatenate(pt_lm),
                point_rawpos=np.concatenate(pt_rawpos),
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[done] {len(pooled)} events ({len(skipped)} skipped) -> "
          f"{args.out}  [{time.time() - t0:.0f}s]")
    if skipped:
        print(f"[skipped] first few: {skipped[:3]}")


if __name__ == "__main__":
    main()
