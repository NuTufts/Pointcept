"""DATA-ISOLATION cell of the deghoster domain study (2026-08-16):
the deployed robust LoRA's EXACT setup — v6-noghosts Sonata encoder + LoRA
recipe (verified recipe-match with the deployed config) — but trained on the
LANTERN (v3-converted) lists instead of prod4 (v2-converted, deleted).

Sharp hypothesis test: the deployed v6+prod4 LoRA transfers to the official
overlay at ~1.0 while every v3-trained model collapses. If THIS cell
collapses on overlay too, the v3 training data carries the domain
difference (conversion pipeline / larmatch-proposal cloud); if it stays
robust, the encoder/head story returns.

Only pretrain path (v6 noghosts), data lists, and save_path differ from
lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py; hooks list
restated (lists merge wholesale).
"""

_base_ = ["./lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py"]

pretrain_model_path = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

save_path = "sonata/lora_deghost_v6noghosts_lantern"

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
