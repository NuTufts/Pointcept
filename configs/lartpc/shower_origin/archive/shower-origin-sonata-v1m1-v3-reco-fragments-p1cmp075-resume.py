"""
Resume variant of shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py.

Inherits the full training/model/dataset config from the cold-start config and
swaps SonataCheckpointLoader for the standard CheckpointLoader. The Sonata
loader prepends "backbone." to every key in the input state_dict, which is
correct only when feeding it a raw SONATA pretrain whose keys start with
"student.backbone.*". Feeding it a ShowerOriginPredictorV3 snapshot (whose
keys are already "backbone.student.backbone.*" / "origin_head.*") produces
"backbone.backbone.*" keys that match nothing, leaving both the backbone and
the origin head randomly initialized on resume.

Usage:
    python tools/train.py \\
        --config configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075-resume.py \\
        --num-gpus 4 \\
        --options resume=True weight=<path/to/model_epoch*.pth>
"""

_base_ = ["./shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py"]

# Replace the cold-start hooks list. CheckpointLoader loads the state_dict
# as-is and, with resume=True, also restores optimizer/scheduler/epoch/scaler.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="AdamStateMonitor", log_frequency=10),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ShowerOriginEvaluator", score_threshold=0.05),
    dict(type="CheckpointSaver", save_freq=None),
]
