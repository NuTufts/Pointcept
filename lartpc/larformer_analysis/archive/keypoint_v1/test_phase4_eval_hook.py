"""Phase 4: validate the LArFormerKeypointEvaluator hook end-to-end.

Drives the real `evaluator.eval()` loop with a minimal mock trainer (no full
training engine): builds the Phase-2 query model from the config, loads
epoch_6, builds a val DataLoader over a few dev-cache events, runs the hook,
and confirms it logs the keypoint metrics — both the auto-accumulated per-pair
errors (val/loss_kpq_*_err_cm) and the PPN per-type recall/precision
(val/kp_R*/val/kp_P*). Recall magnitude isn't the point here (epoch_6 has no
trained keypoint heads); the point is the hook runs + emits the scalars.

    python lartpc/larformer_analysis/archive/keypoint_v1/test_phase4_eval_hook.py
"""

import copy
import sys
import types

import torch
from torch.utils.data import DataLoader

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.larformer import larformer_collate
from pointcept.models.LArFormer.keypoint_particle_evaluator import (
    LArFormerKeypointEvaluator,
)

CONFIG = "configs/lartpc/larformer/stage4_keypoint/archive/larformer-keypoint-query-v1.py"


class _RecWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, k, v, step):
        self.scalars[k] = float(v)


class _Logger:
    def info(self, m):
        print(m)

    def warning(self, m):
        print("WARN:", m)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(CONFIG)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.pop("backbone_weight", None)
    model = build_model(model_cfg).to(device)
    ck = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[hook] loaded {cfg.weight}: missing={len(miss)} unexpected={len(unexp)}")

    ds = build_dataset(cfg.data["val"])
    # Cap to a few events for speed.
    n = min(6, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n)))
    val_loader = DataLoader(subset, batch_size=3, shuffle=False,
                            collate_fn=larformer_collate, num_workers=0)

    writer = _RecWriter()
    trainer = types.SimpleNamespace(
        model=model,
        val_loader=val_loader,
        train_loader=val_loader,
        logger=_Logger(),
        writer=writer,
        epoch=0,
        comm_info={},
        cfg=types.SimpleNamespace(evaluate=True, enable_wandb=False),
    )

    ev = LArFormerKeypointEvaluator(
        coord_scale=float(cfg.get("coord_scale", 179.55)),
        coord_center=cfg.get("coord_center", (125.0, 0.0, 518.0)),
        thresholds_cm=(1.0, 3.0, 10.0), gt_acceptance_cm=5.0,
        empty_cache=False)
    ev.trainer = trainer
    ev.eval()

    sc = writer.scalars
    kp_keys = sorted(k for k in sc if k.startswith("val/kp_"))
    err_keys = sorted(k for k in sc if "kpq" in k and "err_cm" in k)
    print("\n[hook] logged PPN keypoint metrics:")
    for k in kp_keys:
        print(f"   {k} = {sc[k]:.3f}")
    print("[hook] logged per-pair keypoint errors:")
    for k in err_keys:
        print(f"   {k} = {sc[k]:.3f}")

    has_vertex = "val/kp_R1_nu_vertex" in sc
    print(f"[hook] nu_vertex metric present: {has_vertex}  "
          f"res_median={sc.get('val/nu_vertex_res_cm_median')}")
    ok = (len(kp_keys) >= 4              # R/P for >=2 types
          and any("track_start" in k for k in kp_keys)
          and "val/kp_R1_track_start" in sc   # the 1 cm precision target
          and has_vertex                       # dense-head nu vertex logged
          and "current_metric_value" in trainer.comm_info)
    print(f"\n[hook] best_metric published: "
          f"{trainer.comm_info.get('current_metric_name')}="
          f"{trainer.comm_info.get('current_metric_value')}")
    print(f"[hook] {'PASS' if ok else 'FAIL'}: keypoint evaluator logs "
          f"val/kp_* + best_metric during eval")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
