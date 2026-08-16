"""Deghoster-only evaluation — reproduce the training val metrics, then feed
the SAME checkpoint the way the production cascade does, on the same events.

Purpose (2026-08-12 diagnosis): the ft PTv3-decoder deghoster scores
mIoU 0.77 / mAcc 0.89 on its training val, but keeps only ~53% of pi0-photon
charge at tau=0.2 on the run3b data-overlay pilot. Config review found the
two data paths nearly identical (same LogTransform_v6 strength, same coord
normalization to 0.06%; deltas = training-only CenterShift + GridSample
tie-breaking). This tool settles it empirically:

  --mode trainval : build the TRAINING config's data.val pipeline
                    (LArTPCDataset: GridSample -> NormalizeCoord ->
                    LogTransform -> HasmatchAsGhost -> CenterShift) and the
                    model from the same config; expect to REPRODUCE the
                    training-time val mIoU/mAcc on the same val list.
  --mode cascade  : build LArFormerDataset from the CASCADE config's data.test
                    (the exact feed CascadedSlicer passes through to the
                    deghoster: feat = [coord_norm, log-pixval], larformer_
                    collate) and the deghoster from the cascade config's
                    deghoster block + deghoster_weight; same metrics vs
                    hasmatch labels (real=0/ghost=1) + keep-fraction at taus.

If trainval reproduces ~0.77/0.89 and cascade matches it on the same files,
the inference plumbing is validated and the overlay deficit is the model's
domain behavior; a cascade-mode gap instead localizes a setup bug.

    python tools/larformer/run_deghost_eval.py --mode trainval \
        --file-list exp/deghost_ptv3decoder_v2_fullevent_ft/ft_val_2k.txt
    python tools/larformer/run_deghost_eval.py --mode cascade \
        --file-list exp/deghost_ptv3decoder_v2_fullevent_ft/ft_val_2k.txt
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_CANDIDATES = [
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept",
]
for _r in REPO_CANDIDATES:
    if os.path.isdir(os.path.join(_r, "pointcept")):
        sys.path.insert(0, _r)
        break

import pointcept.datasets  # noqa: F401,E402  (register types)
import pointcept.models    # noqa: F401,E402
from pointcept.utils.config import Config              # noqa: E402
from pointcept.datasets.builder import build_dataset   # noqa: E402
from pointcept.models.builder import build_model       # noqa: E402

_KPV2 = REPO_CANDIDATES[0]
TRAIN_CFG = (f"{_KPV2}/configs/lartpc/larformer/stage1_deghost/"
             "deghost-ptv3decoder-v2-fullevent-ft.py")
CASCADE_CFG = (f"{_KPV2}/configs/lartpc/larformer/stage3_particle/"
               "larformer-fullcascade-production-v2-tau020.py")
DEFAULT_WEIGHTS = (f"{_KPV2}/exp/deghost_ptv3decoder_v2_fullevent_ft/"
                   "model/model_best.pth")
TAUS = (0.05, 0.2, 0.35, 0.5)


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> weights {os.path.basename(path)}: "
          f"{len(sd) - len(unexpected)} loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if len(missing) > 10:
        print("    FIRST MISSING:", missing[:5])
    assert len(missing) < len(sd) * 0.5, "majority of weights failed to load"
    return model


class Meter:
    def __init__(self):
        self.i = np.zeros(2); self.u = np.zeros(2); self.t = np.zeros(2)
        self.keep = {t: 0 for t in TAUS}
        self.keep_real = {t: 0 for t in TAUS}
        self.n = 0; self.n_real = 0

    def add(self, pred_cls, label, p_real):
        ok = label >= 0
        pred_cls, label, p_real = pred_cls[ok], label[ok], p_real[ok]
        for c in (0, 1):
            pi, li = pred_cls == c, label == c
            self.i[c] += float(np.sum(pi & li))
            self.u[c] += float(np.sum(pi | li))
            self.t[c] += float(np.sum(li))
        real = label == 0
        self.n += len(label); self.n_real += int(real.sum())
        for t in TAUS:
            k = p_real > t
            self.keep[t] += int(k.sum())
            self.keep_real[t] += int((k & real).sum())

    def report(self, tag):
        iou = self.i / np.maximum(self.u, 1)
        acc = self.i / np.maximum(self.t, 1)
        print(f"\n== {tag} ==  ({self.n} points, {self.n_real} real)")
        print(f"  mIoU {iou.mean():.4f}   mAcc {acc.mean():.4f}   "
              f"allAcc {self.i.sum() / max(self.n, 1):.4f}")
        print(f"  IoU  real {iou[0]:.4f}  ghost {iou[1]:.4f}")
        print(f"  Acc  real {acc[0]:.4f}  ghost {acc[1]:.4f}")
        for t in TAUS:
            print(f"  tau={t:4.2f}: keep-frac {self.keep[t] / max(self.n, 1):.3f}"
                  f"   real-recall {self.keep_real[t] / max(self.n_real, 1):.3f}")


def run_trainval(args, device):
    cfg = Config.fromfile(args.train_config)
    vcfg = dict(cfg.data.val)
    if args.file_list:
        vcfg["data_list_file"] = args.file_list
    dataset = build_dataset(vcfg)
    model = build_model(cfg.model)
    load_weights(model, args.weights)
    model.to(device).eval()
    from pointcept.datasets.utils import collate_fn
    meter = Meter()
    n = len(dataset) if args.max_events < 0 else min(args.max_events,
                                                     len(dataset))
    for i in range(n):
        batch = collate_fn([dataset[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        logits = out["seg_logits"].float()
        p_real = logits.softmax(-1)[:, 0].cpu().numpy()   # HasmatchAsGhost: real=0
        pred = logits.argmax(-1).cpu().numpy()
        label = batch["segment"].cpu().numpy()
        meter.add(pred, label, p_real)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n}")
    meter.report(f"TRAINVAL pipeline ({n} events)")


def run_cascade(args, device):
    cfg = Config.fromfile(args.cascade_config)
    dcfg = dict(cfg.data.test)
    dcfg["data_list_file"] = (os.path.abspath(args.file_list)
                              if args.file_list
                              else dcfg.get("data_list_file"))
    dcfg["max_spacepoints"] = (None if args.max_spacepoints <= 0
                               else args.max_spacepoints)
    if args.lm_threshold is not None:
        dcfg["lm_score_val_threshold"] = float(args.lm_threshold)
    dataset = build_dataset(dcfg)
    assert len(dataset) > 0, (
        f"empty dataset for list {dcfg['data_list_file']} — check the path")
    cs = cfg.model.cascaded_slicer
    model = build_model(dict(cs.deghoster))
    wpath = args.weights or cs.deghoster_weight
    load_weights(model, wpath)
    model.to(device).eval()
    cls_real = int(cs.get("deghoster_class_index_real", 0))
    if args.save_preal:
        os.makedirs(args.save_preal, exist_ok=True)
    from pointcept.datasets.larformer import larformer_collate
    meter = Meter()
    n = len(dataset) if args.max_events < 0 else min(args.max_events,
                                                     len(dataset))
    for i in range(n):
        batch = larformer_collate([dataset[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        logits = out["seg_logits"].float()
        sm = logits.softmax(-1)
        p_real = sm[:, cls_real].cpu().numpy()
        # map to HasmatchAsGhost convention: pred real -> 0, ghost -> 1
        pred = (logits.argmax(-1).cpu().numpy() != cls_real).astype(np.int64)
        label = (batch["hasmatch"].cpu().numpy() == 0).astype(np.int64)
        meter.add(pred, label, p_real)
        if args.save_preal:
            # schema-compatible with the --slice-ids-only sidecars so the
            # photon keep-curve / proximity analyses consume these directly
            import h5py
            with h5py.File(os.path.join(
                    args.save_preal, f"sliceid_event{i:05d}.h5"), "w") as f:
                g = f.create_group("full_slice")
                g.create_dataset(
                    "coord_cm",
                    data=batch["coord"].cpu().numpy().astype(np.float32),
                    compression="gzip")
                g.create_dataset("deghost_p_real",
                                 data=p_real.astype(np.float32),
                                 compression="gzip")
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n}")
    meter.report(f"CASCADE feed ({n} events, class_index_real={cls_real})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=("trainval", "cascade"))
    ap.add_argument("--file-list", default=None,
                    help="merged-h5 list (default: mode's config default)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--train-config", default=TRAIN_CFG,
                    help="training config for trainval mode (model + data.val)")
    ap.add_argument("--cascade-config", default=CASCADE_CFG,
                    help="cascade config whose deghoster block + data.test "
                         "feed cascade mode (use the hybrid config for the "
                         "LoRA model class)")
    ap.add_argument("--save-preal", default=None,
                    help="cascade mode: dir for per-event sliceid_event*.h5 "
                         "(full_slice/{coord_cm,deghost_p_real}) for the "
                         "photon keep-curve / proximity analyses")
    ap.add_argument("--lm-threshold", type=float, default=None,
                    help="cascade mode: override the dataset's lm_score_val_"
                         "threshold (e.g. 0.15 = training-production parity)")
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--max-spacepoints", type=int, default=-1,
                    help="cascade mode cap; <=0 = uncapped")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> mode={args.mode}  device={device}  weights={args.weights}")
    (run_trainval if args.mode == "trainval" else run_cascade)(args, device)


if __name__ == "__main__":
    main()
