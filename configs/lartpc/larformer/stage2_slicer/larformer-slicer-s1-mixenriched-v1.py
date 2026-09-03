"""CELL S1 of SLICER_RETRAIN_PLAN.md (2026-08-19): the data/context +
labels intervention on the m2f-v2 recipe.

Deltas vs larformer-slicer-...-m2frecipe-v2.py (everything else, incl.
the full m2f recipe, inherited):
  A1  train list = MIX v1 (186,529 label-completed overlay events with
      real cosmics/noise + 219,791 label-completed LANTERN enriched;
      46/54; ledger h5list_mix_enriched_train_v1.txt). Val list stays
      LANTERN val (untouched labels — baseline preservation; real
      gating = run_slicer_battery.sh, plan C1).
  A1-loss  masked_no_object=True (enrichment-bar exclusion of unmatched
      queries concentrated on unlabeled points — overlay events have
      unlabeled real cosmics) + mask-DN skipped on events with < 2 GT
      instances.
  A2  labels are completed in-file (r=0.5, shell +-2) on BOTH sources.
  D1  cascade deghoster = v6-lantern ep25 (deployed-LoRA recipe on
      LANTERN data; overlay keep == deployed at matched ga, better
      in-domain — DOMAIN_STUDY_RESULTS.md section 24).
  D2  train-time deghost tau U(0.15, 0.60), val 0.20 — the inherited
      U(0.4, 0.6) centered the OLD chain's 0.5 operating point and left
      tau=0.2 (the v2-era deployment/battery point) OUT of the training
      domain (user catch, 2026-08-19).
Gate: run_slicer_battery.sh per epoch checkpoint (val ref ep4 0.558 /
overlay ref ep4 0.383, old 0.467).
"""

_base_ = ["./larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe-v2.py"]

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"
_LED = f"{_KPV2}/lartpc/data_prep/uboone_official/training_data_ledger"

data = dict(
    train=dict(data_list_file=f"{_LED}/h5list_mix_enriched_train_v1.txt"),
)

model = dict(
    deghoster_weight=(
        f"{_KPV2}/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth"
    ),
    deghost_threshold_min=0.15,
    deghost_threshold_max=0.60,
    deghost_threshold_val=0.20,
    slicer=dict(
        loss_kwargs=dict(
            masked_no_object=True,
        ),
        mask_denoising=dict(
            min_gt_instances=2,
        ),
    ),
)

save_path = "exp/larformer_slicer_s1_mixenriched_v1"
