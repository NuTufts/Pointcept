

import os
import sys
import argparse
import importlib
import importlib.util

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model

from sonata_vis_utils import (
    assign_labels_to_output_points,
    CLASS_NAMES,
    CLASS_COLORS,
    NUM_CLASSES,
    GHOST_LABEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LORA_MODEL_TYPE = "SonataLoRASegmentor"

# v6 coordinate normalisation defaults (used when absent from config)
_V6_COORD_SCALE_DEFAULT  = 1036.0 * (3 ** 0.5) / 2.0 / 5.0
_V6_COORD_CENTER_DEFAULT = [125.0, 0.0, 518.0]

# ColorBrewer Set1 — maximally separated hues for the 8 physics classes.
# Adjust indices here if your class-index mapping differs from the default.
_DISTINCT_COLORS = {
    0: "#E41A1C",   # electron — vivid red
    1: "#377EB8",   # muon     — strong blue
    2: "#FF7F00",   # pion     — vivid orange
    3: "#4DAF4A",   # proton   — bright green
    4: "#984EA3",   # gamma    — strong purple
    5: "#A65628",   # michel   — brown
    6: "#F781BF",   # delta    — pink
    7: "#999999",   # led      — neutral grey
    GHOST_LABEL: "#CCCCCC",  # ghost — light grey
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE visualization of LoRA fine-tuned SONATA (v5/v6 unified)"
    )
    # --- version / paths ---
    p.add_argument("--config-version", type=int, choices=[5, 6], default=6,
        help="Which pretraining convention to use: 5 or 6 (default: 6). "
             "Controls transform pipeline and dataset construction.")
    p.add_argument("--config", type=str, default=None,
        help="Path to LoRA finetune config (.py). Required unless --load-features is set.")
    p.add_argument("--checkpoint", type=str, default=None,
        help="Path to LoRA checkpoint (.pth). Not required with --random-init.")
    p.add_argument("--random-init", action="store_true", default=False,
        help="Use randomly initialised weights (baseline comparison).")
    p.add_argument("--data-list", type=str, default=None,
        help="Override val file list from config.")
    p.add_argument("--output", type=str, default="tsne_lora_unified.png",
        help="Output image path.")

    # --- data collection ---
    p.add_argument("--num-events", type=int, default=50,
        help="Number of validation events to process. Use more events (≥100) "
             "when rare classes (michel, delta, led) need enough real points "
             "to avoid upsampling. Default: 50.")
    p.add_argument("--points-per-event", type=int, default=10000,
        help="Maximum points subsampled per event before accumulation.")
    p.add_argument("--true-points-only", action="store_true", default=False,
        help="Only use true (non-ghost) points.")
    p.add_argument("--label-threshold-factor", type=float, default=4.0,
        help="grid_size multiplier for label assignment threshold.")
    p.add_argument("--grid-size", type=float, default=None,
        help="Override voxel grid size from config.")

    # --- balanced sampling ---
    p.add_argument("--balanced-sampling", action="store_true", default=False,
        help="Class-balanced downsampling before t-SNE. "
             "Sampling is WITHOUT replacement — no duplicate feature vectors. "
             "Classes with fewer real points than --points-per-class are "
             "capped at their real count and a warning is printed. "
             "Increase --num-events to collect more real points.")
    p.add_argument("--points-per-class", type=int, default=2000,
        help="Target points per class for balanced sampling (no-duplication). "
             "Default: 2000. Reduce if rare classes are still too small.")
    p.add_argument("--max-points", type=int, default=50000,
        help="Max total points for unbalanced random subsampling "
             "(only used when --balanced-sampling is NOT set).")

    # --- t-SNE ---
    p.add_argument("--perplexity", type=float, default=100.0,
        help="t-SNE perplexity. Higher values (100–200) give smoother, more "
             "globally-aware layouts and reduce isolated-halo artifacts. "
             "Default: 100.")
    p.add_argument("--learning-rate", type=float, default=200.0,
        help="t-SNE learning rate. Default: 200.")
    p.add_argument("--n-iter", type=int, default=2000,
        help="t-SNE iterations. 2000 is safer for convergence with large N. "
             "Default: 2000.")
    p.add_argument("--early-exaggeration", type=float, default=8.0,
        help="t-SNE early exaggeration factor. Lower values (6–8) reduce "
             "the risk of tight-cluster lock-in that creates halos around "
             "rare classes. Default: 8.")
    p.add_argument("--normalize-features", action="store_true", default=False,
        help="Apply StandardScaler whitening to features before t-SNE so "
             "high-variance backbone dimensions do not dominate distances.")

    # --- misc ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-features", type=str, default=None,
        help="Save extracted features to .npz for later re-plotting.")
    p.add_argument("--load-features", type=str, default=None,
        help="Load pre-extracted features from .npz (skips all inference).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Balanced sampling — NO duplication
# ---------------------------------------------------------------------------

def balanced_subsample(features, labels, points_per_class, num_classes,
                       coords=None, seed=42):
    """Sample up to ``points_per_class`` real points per class, WITHOUT replacement.

    Unlike the original scripts this function never duplicates feature vectors.
    If a class has fewer real points than ``points_per_class`` it is capped at
    its actual count and a warning is printed. The t-SNE plot will then show
    unequal class sizes for rare classes, but will be free of the circular
    halo artifacts caused by repeated identical embeddings.

    To get more points for rare classes, run more events (--num-events).
    """
    np.random.seed(seed)
    unique_labels = np.unique(labels)
    selected_indices = []

    print(f"\n  Balanced sampling (NO duplication), target={points_per_class}/class:")
    for cls_idx in unique_labels:
        cls_indices = np.where(labels == cls_idx)[0]
        n_real = len(cls_indices)
        n_take = min(n_real, points_per_class)
        if n_real < points_per_class:
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            print(f"    WARNING class {cls_idx} ({cls_name}): only {n_real} real points "
                  f"< target {points_per_class}. Taking all {n_real}. "
                  f"Run more --num-events to fix this.")
        else:
            print(f"    Class {cls_idx}: {n_take}/{n_real} points")
        sampled = np.random.choice(cls_indices, n_take, replace=False)
        selected_indices.append(sampled)

    selected_indices = np.concatenate(selected_indices)
    np.random.shuffle(selected_indices)
    print(f"  Total after balanced sampling: {len(selected_indices)} points\n")

    features_out = features[selected_indices]
    labels_out   = labels[selected_indices]
    coords_out   = coords[selected_indices] if coords is not None else None
    return features_out, labels_out, coords_out


# ---------------------------------------------------------------------------
# t-SNE backend (cuML → sklearn → pure-numpy fallback)
# ---------------------------------------------------------------------------

class SimpleTSNE:
    """Pure-numpy exact t-SNE. Fallback when cuML and sklearn are unavailable
    (e.g. P100 nodes where cuML requires CC≥7.0).
    Suitable for up to ~50 k points; expect 5–20 min for n_iter=2000.
    """
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, random_state=42):
        self.n_components       = n_components
        self.perplexity         = perplexity
        self.learning_rate      = learning_rate
        self.n_iter             = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state       = random_state

    @staticmethod
    def _pairwise_sq_dists(X):
        sq = np.sum(X ** 2, axis=1)
        D  = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.clip(D, 0, None, out=D)
        return D

    @staticmethod
    def _compute_P(D, perplexity):
        n = D.shape[0]
        P = np.zeros((n, n), dtype=np.float32)
        log_perp = np.log(perplexity)
        for i in range(n):
            di = D[i].copy()
            di[i] = np.inf
            beta_lo, beta_hi = -np.inf, np.inf
            beta = 1.0
            for _ in range(50):
                e = np.exp(-di * beta)
                e[i] = 0.0
                s = e.sum() + 1e-10
                H = np.log(s) + beta * np.sum(di * e) / s
                if abs(H - log_perp) < 1e-5:
                    break
                if H > log_perp:
                    beta_lo = beta
                    beta = beta * 2 if beta_hi == np.inf else (beta + beta_hi) / 2
                else:
                    beta_hi = beta
                    beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2
            e = np.exp(-di * beta); e[i] = 0.0
            P[i] = e / (e.sum() + 1e-10)
        P = (P + P.T) / (2 * n)
        np.clip(P, 1e-12, None, out=P)
        return P

    def fit_transform(self, X):
        np.random.seed(self.random_state)
        n = X.shape[0]
        print(f"  [SimpleTSNE] n={n}, perplexity={self.perplexity}, "
              f"lr={self.learning_rate}, n_iter={self.n_iter}")
        X = X - X.mean(axis=0)
        if X.shape[1] > 50:
            print("  [SimpleTSNE] PCA pre-reduction to 50 dims...")
            cov = (X.T @ X) / (n - 1)
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argsort(vals)[::-1][:50]
            X = X @ vecs[:, idx]

        print("  [SimpleTSNE] Computing pairwise distances...")
        D = self._pairwise_sq_dists(X.astype(np.float32))
        print("  [SimpleTSNE] Computing P matrix...")
        P = self._compute_P(D, self.perplexity)
        del D

        Y      = np.random.randn(n, self.n_components).astype(np.float32) * 0.0001
        dY     = np.zeros_like(Y)
        gains  = np.ones_like(Y)
        momentum = 0.5

        print(f"  [SimpleTSNE] Optimising for {self.n_iter} iterations...")
        for t in range(1, self.n_iter + 1):
            Pt = P * self.early_exaggeration if t <= 250 else P
            if t == 251:
                momentum = 0.8
            sq_Y   = np.sum(Y ** 2, axis=1)
            dist_Y = sq_Y[:, None] + sq_Y[None, :] - 2.0 * (Y @ Y.T)
            np.clip(dist_Y, 0, None, out=dist_Y)
            Q = 1.0 / (1.0 + dist_Y)
            np.fill_diagonal(Q, 0.0)
            Q_norm = np.maximum(Q / (Q.sum() + 1e-10), 1e-12)
            PQ   = (Pt - Q_norm) * Q
            grad = np.zeros_like(Y)
            for i in range(n):
                diff    = Y[i] - Y
                grad[i] = 4.0 * (PQ[i, :, None] * diff).sum(axis=0)
            gains = (gains + 0.2) * ((grad > 0) != (dY > 0)) + \
                    (gains * 0.8) * ((grad > 0) == (dY > 0))
            np.clip(gains, 0.01, None, out=gains)
            dY  = momentum * dY - self.learning_rate * gains * grad
            Y  += dY
            Y  -= Y.mean(axis=0)
            if t % 100 == 0:
                kl = np.sum(Pt * np.log(np.maximum(Pt, 1e-12) /
                                        np.maximum(Q_norm, 1e-12)))
                print(f"    iter {t:4d}  KL={kl:.4f}")
        return Y


def get_tsne_reducer(perplexity, learning_rate, n_iter, early_exaggeration,
                     random_state):
    """Return a t-SNE object with fit_transform(X).
    Priority: cuML (GPU) → sklearn (CPU) → SimpleTSNE (numpy fallback).
    """
    try:
        from cuml.manifold import TSNE as cumlTSNE
        print("Using cuML (GPU) t-SNE")
        return cumlTSNE(n_components=2, perplexity=perplexity,
                        learning_rate=learning_rate, n_iter=n_iter,
                        early_exaggeration=early_exaggeration,
                        random_state=random_state, verbose=1)
    except Exception:
        pass

    try:
        from sklearn.manifold import TSNE as sklearnTSNE
        import sklearn
        print(f"Using sklearn (CPU) t-SNE  (sklearn {sklearn.__version__})")
        ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        iter_kw = "max_iter" if ver >= (1, 5) else "n_iter"
        return sklearnTSNE(n_components=2, perplexity=perplexity,
                           learning_rate=learning_rate,
                           **{iter_kw: n_iter},
                           early_exaggeration=early_exaggeration,
                           random_state=random_state, verbose=1)
    except ImportError:
        pass

    print("Using pure-numpy SimpleTSNE fallback "
          "(cuML unavailable on CC=6.0, sklearn not installed)")
    return SimpleTSNE(n_components=2, perplexity=perplexity,
                      learning_rate=learning_rate, n_iter=n_iter,
                      early_exaggeration=early_exaggeration,
                      random_state=random_state)


# ---------------------------------------------------------------------------
# Transform pipelines — one per config version
# ---------------------------------------------------------------------------

def build_inference_transform_v5(cfg, grid_size):
    """v5 inference transform.

    Matches the v5 val pipeline: BiasedSphereCrop → GridSample → ToTensor → Collect.
    feat_keys = ("strength",)  — v5 uses only energy deposition, not coord.

    NOTE: The original v5 visualisation script had these reversed (GridSample
    before BiasedSphereCrop). This version is correct.
    """
    return [
        # 1. Crop first — then voxelise the cropped region only.
        dict(
            type="BiasedSphereCrop",
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=20.0,
            point_max=20480,
            point_min=4096,
            prob_random=0.25,
            max_retries=100,
            fallback_to_random=True,
        ),
        # 2. Voxelise
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        # 3. Log-transform ADC (replaces log_transform_edep dataset flag)
        dict(
            type="LogTransform",
            min_val=0.01,
            max_val=1000.0,
            log=True,
            keys=("strength",),
        ),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "name", "grid_size"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("strength",),
        ),
    ]


def build_inference_transform_v6(cfg, grid_size):
    """v6 inference transform.

    Matches the v6 val pipeline exactly:
      BiasedSphereCrop → GridSample → NormalizeCoord → LogTransform(max=1000)
      → ToTensor → Collect
    feat_keys = ("coord", "strength") — coord channels first, then strength.

    max_val=1000.0 (corrected from 500.0 in v6_2 which shifted the
    log-normalised distribution at eval time).
    """
    coord_scale  = cfg.get("coord_scale",  _V6_COORD_SCALE_DEFAULT)
    coord_center = cfg.get("coord_center", _V6_COORD_CENTER_DEFAULT)
    print(f"  [v6 transform] coord_scale={coord_scale:.6f}, "
          f"coord_center={coord_center}")
    return [
        # 1. Crop
        dict(
            type="BiasedSphereCrop",
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=20.0,
            point_max=20480,
            point_min=4096,
            prob_random=0.25,
            max_retries=100,
            fallback_to_random=True,
        ),
        # 2. Voxelise
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        # 3. Normalise coordinates — must match v6 pretraining exactly
        dict(
            type="NormalizeCoord",
            center=coord_center,
            scale=coord_scale,
        ),
        # 4. Log-transform ADC — max_val=1000.0 matches v6 pretraining
        dict(
            type="LogTransform",
            min_val=0.01,
            max_val=1000.0,
            log=True,
            keys=("strength",),
        ),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "name", "grid_size"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("coord", "strength"),
        ),
    ]


def build_inference_transform(cfg, grid_size, version):
    if version == 5:
        return build_inference_transform_v5(cfg, grid_size)
    elif version == 6:
        return build_inference_transform_v6(cfg, grid_size)
    else:
        raise ValueError(f"Unknown config version: {version}")


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch, in_channels=None):
    coords, feats, segments, names, offsets = [], [], [], [], []
    offset = 0
    for data in batch:
        n = data["coord"].shape[0]
        coords.append(data["coord"])
        feat = data["feat"][:, :in_channels] if in_channels is not None else data["feat"]
        feats.append(feat)
        if "segment" in data:
            segments.append(data["segment"])
        names.append(data.get("name", "unknown"))
        offset += n
        offsets.append(offset)
    result = {
        "coord":  torch.cat(coords,  dim=0),
        "feat":   torch.cat(feats,   dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "name":   names,
    }
    if segments:
        result["segment"] = torch.cat(segments, dim=0)
    if "grid_coord" in batch[0]:
        result["grid_coord"] = torch.cat([d["grid_coord"] for d in batch], dim=0)
    if "grid_size" in batch[0]:
        result["grid_size"] = batch[0]["grid_size"]
    return result


# ---------------------------------------------------------------------------
# LoRA model loading
# ---------------------------------------------------------------------------

def _is_lora_config(cfg):
    return getattr(cfg.model, "type", "") == LORA_MODEL_TYPE


def _register_lora_model():
    """Import lora_sonata so SonataLoRASegmentor is in the MODELS registry."""
    for candidate in ["lora_sonata", "tools.lora_sonata",
                      "pointcept.models.lora_sonata"]:
        try:
            importlib.import_module(candidate)
            print(f"  Registered SonataLoRASegmentor via import '{candidate}'")
            return
        except ModuleNotFoundError:
            pass

    here      = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    search_paths = [
        "/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept/pointcept/models/lora_sonata.py",
        os.path.join(here,      "lora_sonata.py"),
        os.path.join(repo_root, "lora_sonata.py"),
        os.path.join(repo_root, "pointcept", "models", "lora_sonata.py"),
        os.path.join(repo_root, "tools", "lora_sonata.py"),
    ]
    for path in search_paths:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("lora_sonata", path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"  Registered SonataLoRASegmentor from {path}")
            return

    searched = "\n".join(f"  {p}" for p in search_paths)
    raise ImportError(
        f"Cannot find lora_sonata.py.\nSearched:\n{searched}\n"
        "Fix: export PYTHONPATH=/path/to/dir/containing/lora_sonata:$PYTHONPATH"
    )


def load_model(cfg, checkpoint_path, device, random_init=False):
    is_lora = _is_lora_config(cfg)
    print(f"Building model: {cfg.model.type}  (LoRA={is_lora})")

    if is_lora:
        _register_lora_model()
        cfg.model.backbone.up_cast_level = 4
    else:
        cfg.model.up_cast_level = 4

    model = build_model(cfg.model)

    if random_init:
        print("Using RANDOMLY INITIALISED weights.")
    else:
        if checkpoint_path is None:
            raise ValueError("--checkpoint required unless --random-init is set.")
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Strip DDP prefix
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        info = model.load_state_dict(state_dict, strict=False)
        if is_lora:
            lora_miss  = [k for k in info.missing_keys if "lora_"    in k]
            head_miss  = [k for k in info.missing_keys if "seg_head"  in k]
            other_miss = [k for k in info.missing_keys
                          if k not in lora_miss and k not in head_miss]
            print(f"  LoRA adapter keys missing (expected for pretrain ckpt): {len(lora_miss)}")
            print(f"  seg_head keys missing (expected for pretrain ckpt):     {len(head_miss)}")
            if other_miss:
                print(f"  WARNING — other missing keys: {other_miss[:10]}")
        else:
            if info.missing_keys:
                print(f"  Missing keys: {info.missing_keys[:10]}")
        if info.unexpected_keys:
            print(f"  Unexpected keys: {info.unexpected_keys[:10]}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


def get_in_channels(cfg):
    """Resolve in_channels, handling the doubly-nested LoRA backbone config."""
    if _is_lora_config(cfg):
        return cfg.model.backbone.backbone.in_channels
    return cfg.model.backbone.in_channels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, data_dict, device):
    """Extract encoder features, routing through backbone for LoRA models."""
    inp = {
        "coord":  data_dict["coord"].to(device),
        "feat":   data_dict["feat"].to(device),
        "offset": data_dict["offset"].to(device),
    }
    print(f"  feat.shape={inp['feat'].shape}", end="  ")

    gs = data_dict.get("grid_size", 0.25)
    inp["grid_size"] = (
        gs[0].item() if isinstance(gs, torch.Tensor) and gs.numel() > 1
        else gs.item() if isinstance(gs, torch.Tensor)
        else gs
    )
    if "grid_coord" in data_dict:
        inp["grid_coord"] = data_dict["grid_coord"].to(device)

    # LoRA: route through model.backbone so adapters are applied
    if type(model).__name__ == "SonataLoRASegmentor":
        result = model.backbone.forward(inp, return_point=True)
    else:
        result = model.forward(inp, return_point=True)

    return result["point"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class_names = CLASS_NAMES
    num_classes  = NUM_CLASSES

    print(f"CLASS_NAMES: {CLASS_NAMES}")
    print(f"Config version: v{args.config_version}")

    # ------------------------------------------------------------------
    # Load pre-extracted features OR run inference
    # ------------------------------------------------------------------
    if args.load_features is not None:
        print(f"\nLoading pre-extracted features from: {args.load_features}")
        npz          = np.load(args.load_features)
        all_features = npz["features"]
        all_labels   = npz["labels"]
        all_coords   = npz.get("coords", None)
        print(f"  Loaded {len(all_features)} points, {all_features.shape[1]}D features")

    else:
        if args.config is None:
            raise ValueError("--config is required unless --load-features is set.")

        print(f"\nLoading config: {args.config}")
        cfg = Config.fromfile(args.config)

        grid_size = args.grid_size if args.grid_size is not None \
                    else cfg.get("grid_size", 0.25)
        print(f"Grid size: {grid_size}")

        transform = build_inference_transform(cfg, grid_size, args.config_version)

        # --- resolve data list ---
        data_list = args.data_list
        if data_list is None:
            if hasattr(cfg, "VAL_FILE_LIST"):
                data_list = cfg.VAL_FILE_LIST
            elif (hasattr(cfg, "data") and hasattr(cfg.data, "val")
                    and "data_list_file" in cfg.data.val):
                data_list = cfg.data.val.data_list_file
            elif hasattr(cfg, "TRAIN_FILE_LIST"):
                print("Warning: no val file list found, falling back to TRAIN_FILE_LIST")
                data_list = cfg.TRAIN_FILE_LIST
            else:
                raise ValueError(
                    "Cannot determine data list. Pass --data-list or set "
                    "VAL_FILE_LIST / data.val.data_list_file in your config."
                )
        print(f"Data list: {data_list}")

        # --- build dataset ---
        from pointcept.datasets.lartpc import LArTPCDataset
        dataset_kwargs = dict(
            split="val",
            data_list_file=data_list,
            transform=transform,
            use_reco_coords=True,
            use_edep_as_strength=True,
            label_mode="ssnet",
            coord_scale=1.0,
            include_ghosts=(args.config_version == 5),   # v5 includes ghosts, v6 does not
            exclude_other=True,
            true_points_only=args.true_points_only,
            test_mode=False,
            loop=1,
        )
        if args.config_version == 5:
            # v5 used a dataset-level flag; v6 handles this via the transform pipeline
            dataset_kwargs["log_transform_edep"] = True
        dataset = LArTPCDataset(**dataset_kwargs)
        print(f"Dataset size: {len(dataset)} events")

        # --- model ---
        model      = load_model(cfg, args.checkpoint, args.device,
                                random_init=args.random_init)
        in_channels = get_in_channels(cfg)
        print(f"Model expects {in_channels} input feature channels")

        all_features, all_labels, all_coords = [], [], []
        num_events = min(args.num_events, len(dataset))
        print(f"\nExtracting features from {num_events} events...")

        for event_idx in range(num_events):
            print(f"\n  Event {event_idx + 1}/{num_events} ...", end=" ")
            data       = dataset[event_idx]
            batch_data = collate_fn([data], in_channels=in_channels)
            n_in       = batch_data["coord"].shape[0]
            print(f"{n_in} input points", end="  ")

            point    = extract_features(model, batch_data, args.device)
            features = point.feat.cpu().numpy()
            coords   = point.coord.cpu().numpy()
            n_out    = features.shape[0]
            print(f"-> {n_out} output points, {features.shape[1]}D features")

            # Label assignment
            if "segment" in batch_data:
                labels = assign_labels_to_output_points(
                    output_coords=coords,
                    input_coords=batch_data["coord"].cpu().numpy(),
                    input_labels=batch_data["segment"].cpu().numpy(),
                    grid_size=grid_size,
                    threshold_factor=args.label_threshold_factor,
                    ghost_label=GHOST_LABEL,
                    use_gpu=False,
                    verbose=False,
                )
            else:
                labels = np.zeros(n_out, dtype=np.int64)

            # Per-event cap (no replacement)
            if n_out > args.points_per_event:
                idx      = np.random.choice(n_out, args.points_per_event, replace=False)
                features = features[idx]
                labels   = labels[idx]
                coords   = coords[idx]

            all_features.append(features)
            all_labels.append(labels)
            all_coords.append(coords)

        all_features = np.vstack(all_features)
        all_labels   = np.concatenate(all_labels)
        all_coords   = np.vstack(all_coords)
        print(f"\nTotal accumulated: {len(all_features)} points, "
              f"{all_features.shape[1]}D features")

        if args.save_features is not None:
            print(f"Saving features to: {args.save_features}")
            np.savez(args.save_features,
                     features=all_features, labels=all_labels, coords=all_coords)

    # ------------------------------------------------------------------
    # Subsampling for t-SNE
    # ------------------------------------------------------------------
    print("\nClass distribution before subsampling:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for ci, cnt in zip(unique, counts):
        name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
        print(f"  {name}: {cnt} ({100*cnt/len(all_labels):.1f}%)")

    if args.balanced_sampling:
        all_features, all_labels, all_coords = balanced_subsample(
            all_features, all_labels,
            points_per_class=args.points_per_class,
            num_classes=num_classes,
            coords=all_coords,
            seed=args.seed,
        )
    elif len(all_features) > args.max_points:
        print(f"\nRandom subsampling {len(all_features)} → {args.max_points} points")
        idx          = np.random.choice(len(all_features), args.max_points, replace=False)
        all_features = all_features[idx]
        all_labels   = all_labels[idx]
        if all_coords is not None:
            all_coords = all_coords[idx]

    print(f"\nFinal point count for t-SNE: {len(all_features)}")

    # ------------------------------------------------------------------
    # Optional feature normalisation
    # ------------------------------------------------------------------
    if args.normalize_features:
        print("Applying StandardScaler whitening to features...")
        from sklearn.preprocessing import StandardScaler
        all_features = StandardScaler().fit_transform(all_features).astype(np.float32)

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    print(f"\nRunning t-SNE:  perplexity={args.perplexity}  "
          f"lr={args.learning_rate}  n_iter={args.n_iter}  "
          f"early_exaggeration={args.early_exaggeration}")
    reducer   = get_tsne_reducer(args.perplexity, args.learning_rate,
                                  args.n_iter, args.early_exaggeration, args.seed)
    embedding = reducer.fit_transform(all_features.astype(np.float32))
    print(f"t-SNE complete: embedding shape = {embedding.shape}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    class_colors = {**CLASS_COLORS, **_DISTINCT_COLORS}
    ghost_idx    = GHOST_LABEL
    init_label   = "RANDOM INIT" if args.random_init else f"LoRA v{args.config_version}"

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"SONATA t-SNE  [{init_label}]  "
                 f"perplexity={args.perplexity}  n_iter={args.n_iter}  "
                 f"early_exag={args.early_exaggeration}",
                 fontsize=11)

    # --- Panel 1: all points, ghost faded ---
    ax1 = axes[0]
    ghost_mask = all_labels == ghost_idx
    if ghost_mask.sum() > 0:
        ax1.scatter(embedding[ghost_mask, 0], embedding[ghost_mask, 1],
                    c=class_colors.get(ghost_idx, "#CCCCCC"),
                    label=f"ghost ({ghost_mask.sum()})",
                    s=2, alpha=0.10, rasterized=True)
    for ci in range(num_classes):
        if ci == ghost_idx:
            continue
        mask = all_labels == ci
        if mask.sum() == 0:
            continue
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=class_colors.get(ci, "#000000"),
                    label=f"{class_names[ci]} ({mask.sum()})",
                    s=2, alpha=0.6, rasterized=True)
    ax1.set_title("All points  (ghost α=0.10)")
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
    ax1.legend(markerscale=5, fontsize=7, loc="best")

    # --- Panel 2: true points only ---
    ax2 = axes[1]
    for ci in range(num_classes):
        if ci == ghost_idx:
            continue
        mask = all_labels == ci
        if mask.sum() == 0:
            continue
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=class_colors.get(ci, "#000000"),
                    label=f"{class_names[ci]} ({mask.sum()})",
                    s=3, alpha=0.7, rasterized=True)
    non_ghost = (all_labels != ghost_idx).sum()
    ax2.set_title(f"True points only  ({non_ghost} pts)")
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.legend(markerscale=5, fontsize=7, loc="best")

    # --- Panel 3: density hexbin ---
    ax3 = axes[2]
    hb = ax3.hexbin(embedding[:, 0], embedding[:, 1],
                    gridsize=60, cmap="viridis", mincnt=1)
    ax3.set_title(f"Point density  ({len(embedding)} total)")
    ax3.set_xlabel("t-SNE 1"); ax3.set_ylabel("t-SNE 2")
    plt.colorbar(hb, ax=ax3, label="Count")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {args.output}")

    # Save embedding for re-plotting without re-running inference or t-SNE
    emb_path = args.output.rsplit(".", 1)[0] + "_embedding.npz"
    np.savez(emb_path,
             embedding=embedding,
             labels=all_labels,
             class_names=class_names,
             perplexity=args.perplexity,
             learning_rate=args.learning_rate,
             n_iter=args.n_iter,
             early_exaggeration=args.early_exaggeration,
             config_version=args.config_version)
    print(f"Saved embedding to: {emb_path}")

    plt.show()


if __name__ == "__main__":
    main()
