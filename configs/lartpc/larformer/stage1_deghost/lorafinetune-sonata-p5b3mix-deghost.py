"""LoRA deghoster on the P5B.3 MIXED sim+data Sonata encoder — the
low-capacity arm of the encoder-swap A/B (2026-08-14).

Pairs with deghost-ptv3decoder-p5b3mix-v1.py to complete the 2x2
{encoder: v7 data-only, P5B.3 sim+data mix} x {head: LoRA+linear,
PTv3 decoder}: the existing LoRA-on-v7 and ft-decoder-on-v7 checkpoints are
the other two cells. Judged on BOTH domains (corsika val + overlay photon
keep-curve) — the v7 cells measured: LoRA 0.854 / decoder 0.531 photon keep
@ tau=0.2 on overlay.

Identical recipe to lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-
larmatch.py EXCEPT: (1) the pretrained encoder path; (2) the training data —
the original prod4 v2_expandedclasses lists are DELETED from disk, so this
retrains on the v3 LANTERN merged-h5 lists (the same data the PTv3-decoder
arm uses; noted A/B deviation for the LoRA ROW: the LoRA-on-v7 baseline was
prod4-trained). Config lists merge WHOLESALE, so the hooks list is restated
with the new pretrained_path.
"""

_base_ = ["./lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py"]

pretrain_model_path = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept/"
    "sonata/p5b/P5B.3-mix_larmatch-s0/model/epoch_18.pth"
)

save_path = "sonata/lora_deghost_p5b3mix_hasmatch"

_LANTERN = ("/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
            "lartpc_data_prep/lantern_scripts/h5lists")
data = dict(
    train=dict(data_list_file=f"{_LANTERN}/h5list_mcall_lantern_train.txt"),
    val=dict(data_list_file=f"{_LANTERN}/h5list_mcall_lantern_val.txt"),
    test=dict(data_list_file=f"{_LANTERN}/h5list_mcall_lantern_val.txt"),
)

hooks = [
    dict(
        type="LoRASonataCheckpointLoader",
        pretrained_path=pretrain_model_path,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
]
