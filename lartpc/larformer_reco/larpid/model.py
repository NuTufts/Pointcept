"""LArPID model wrapper: the prongCNN 3-plane shared-weight ResNet-34 with the
production preprocessing (per-channel Normalize + clamp(max=4)) and the
official MicroBooNE checkpoints.

Checkpoint selection (train-data-leakage rule): files/samples with 'run3' in
their tag use LArPID_alternate_network_weights.pt (trained on run-1 MC); all
other runs use LArPID_default_network_weights.pt (trained on run-3 MC).

Heads: 5-class PID log-softmax (0=e,1=gamma,2=mu,3=pi,4=p), completeness,
purity, 3-class process log-softmax (0=primary, 1=from neutral parent,
2=from charged parent).
"""
import os
import sys

import numpy as np
import torch

PRONGCNN_DIR = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/prongCNN"
CHECKPOINT_DIR = os.path.join(PRONGCNN_DIR, "checkpoints")
PID_PDG = (11, 22, 13, 211, 2212)


def select_checkpoint(sample_tag, checkpoint_dir=CHECKPOINT_DIR):
    """'run3' anywhere in the tag -> alternate weights; else default."""
    name = ("LArPID_alternate_network_weights.pt" if "run3" in sample_tag
            else "LArPID_default_network_weights.pt")
    return os.path.join(checkpoint_dir, name)


class LArPID:
    def __init__(self, checkpoint, device="cuda"):
        sys.path.insert(0, os.path.join(PRONGCNN_DIR, "models"))
        from models_instanceNorm_reco_2chan_quadTask import ResBlock, ResNet34
        from normalization_constants import mean, std
        self.device = device
        self.model = ResNet34(2, ResBlock, outputs=5)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        if all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.mean = torch.tensor(mean, dtype=torch.float32,
                                 device=device).view(1, 6, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32,
                                device=device).view(1, 6, 1, 1)
        self.checkpoint = os.path.basename(checkpoint)

    @torch.no_grad()
    def __call__(self, images):
        """images: (B, 6, 512, 512) float32 numpy (raw ADC crops).
        Returns dict of numpy arrays: class_scores (B,5) log-softmax,
        completeness (B,), purity (B,), process_scores (B,3) log-softmax,
        pid (B,) argmax PDG, process (B,) argmax code."""
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        x = torch.clamp((x - self.mean) / self.std, max=4.0)
        cls, comp, pur, proc = self.model(x)
        cls = cls.cpu().numpy()
        proc = proc.cpu().numpy()
        return {
            "class_scores": cls.astype(np.float32),
            "completeness": comp.reshape(-1).cpu().numpy().astype(np.float32),
            "purity": pur.reshape(-1).cpu().numpy().astype(np.float32),
            "process_scores": proc.astype(np.float32),
            "pid": np.asarray([PID_PDG[i] for i in cls.argmax(1)], np.int32),
            "process": proc.argmax(1).astype(np.int32),
        }
