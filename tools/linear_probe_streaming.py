"""
Streaming Linear Probing Evaluation Tool for Sonata Pretrained Models

This tool evaluates learned representations using online/streaming SGD.
Features are extracted on-the-fly for each mini-batch and immediately
discarded after use, enabling evaluation on arbitrarily large datasets
with constant memory usage.

Trade-off: Slower (re-extracts features each epoch) but very memory efficient.

Usage:
    python tools/linear_probe_streaming.py --config-file CONFIG --weight WEIGHT \\
        --train-list TRAIN.txt --test-list TEST.txt

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Sequence
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset


# ============================================================================
# Linear Classifier
# ============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier for probing."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================================
# Streaming Linear Probe Trainer
# ============================================================================

class StreamingLinearProbeTrainer:
    """
    Train linear classifiers using streaming/online SGD.

    Features are extracted on-the-fly and discarded after each batch,
    enabling constant memory usage regardless of dataset size.
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        num_classes: int,
        feature_dim: int,
        learning_rates: Sequence[float],
        epochs: int = 10,
        weight_decay: float = 0.01,
        use_teacher: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.use_teacher = use_teacher
        self.device = device

        # Build classifiers for each learning rate
        self.classifiers = {}
        self.optimizers = {}

        for lr in learning_rates:
            name = f"lr_{lr:.6f}".replace(".", "_")
            clf = LinearClassifier(feature_dim, num_classes).to(device)
            opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
            self.classifiers[name] = clf
            self.optimizers[name] = opt

        self.criterion = nn.CrossEntropyLoss()

    def _extract_features_batch(self, data_dict):
        """Extract features for a single batch, returning features and labels."""
        # Save segment labels before backbone processing
        segment_labels = data_dict['segment'].clone()

        # Move data to device
        for key in list(data_dict.keys()):
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device)
            elif key == 'grid_size' and not isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = torch.tensor(data_dict[key], device=self.device)

        # Get backbone
        network = self.model.teacher if self.use_teacher else self.model.student
        backbone = network['backbone']

        # Run backbone
        point = backbone(data_dict)

        # Upcast features through decoder levels
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        # Get inverse mapping
        if "serialized_inverse" in point.keys():
            inverse = point.serialized_inverse[0]
        else:
            inverse = torch.arange(point.feat.shape[0], device=point.feat.device)

        # Get features at original resolution
        features = point.feat[inverse]  # [N, C]
        labels = segment_labels.squeeze(-1) if segment_labels.dim() > 1 else segment_labels
        labels = labels.to(self.device)

        # Filter out ignore_index (-1)
        valid_mask = labels >= 0
        features = features[valid_mask]
        labels = labels[valid_mask]

        return features, labels

    def train_epoch(self, epoch: int):
        """Train all classifiers for one epoch using streaming features."""
        for clf in self.classifiers.values():
            clf.train()

        total_loss = {name: 0.0 for name in self.classifiers}
        total_samples = 0

        self.model.eval()  # Backbone in eval mode

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

        for data_dict in pbar:
            with torch.no_grad():
                features, labels = self._extract_features_batch(data_dict)

            if len(labels) == 0:
                continue

            batch_size = len(labels)
            total_samples += batch_size

            # Train each classifier
            for name, clf in self.classifiers.items():
                self.optimizers[name].zero_grad()
                logits = clf(features)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizers[name].step()

                total_loss[name] += loss.item() * batch_size

            # Update progress bar
            avg_loss = sum(total_loss.values()) / (len(self.classifiers) * max(total_samples, 1))
            pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

        # Print epoch summary
        if total_samples > 0:
            print(f"  Epoch {epoch+1}/{self.epochs} - ", end="")
            for name in sorted(self.classifiers.keys())[:3]:  # Show first 3
                avg = total_loss[name] / total_samples
                print(f"{name}: {avg:.4f}  ", end="")
            print(f"... ({total_samples:,} samples)")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, object]:
        """Evaluate all classifiers using streaming features."""
        for clf in self.classifiers.values():
            clf.eval()
        self.model.eval()

        # Accumulators for each classifier
        all_preds = {name: [] for name in self.classifiers}
        all_labels = []

        print("Evaluating...")
        for data_dict in tqdm(self.test_loader, desc="Evaluating", leave=False):
            with torch.no_grad():
                features, labels = self._extract_features_batch(data_dict)

            if len(labels) == 0:
                continue

            all_labels.append(labels.cpu())

            for name, clf in self.classifiers.items():
                logits = clf(features)
                preds = logits.argmax(dim=-1)
                all_preds[name].append(preds.cpu())

        if not all_labels:
            return {"error": "No valid test samples"}

        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels).numpy()
        for name in all_preds:
            all_preds[name] = torch.cat(all_preds[name]).numpy()

        # Compute metrics for each classifier
        results = {}
        best_name = ""
        best_mf1 = -1.0

        for name, preds in all_preds.items():
            metrics = self._compute_metrics(preds, all_labels)
            results[name] = metrics
            if metrics['m_f1'] > best_mf1:
                best_mf1 = metrics['m_f1']
                best_name = name

        return {
            'best_classifier': best_name,
            'best_mf1': best_mf1,
            'all_results': results,
            'best_metrics': results[best_name],
            'labels': all_labels,
            'best_preds': all_preds[best_name],
        }

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
        """Compute classification metrics."""
        num_classes = self.num_classes

        precision_class = np.zeros(num_classes, dtype=np.float64)
        recall_class = np.zeros(num_classes, dtype=np.float64)
        f1_class = np.zeros(num_classes, dtype=np.float64)
        iou_class = np.zeros(num_classes, dtype=np.float64)

        for c in range(num_classes):
            pred_c = preds == c
            gt_c = labels == c

            tp = np.logical_and(pred_c, gt_c).sum()
            fp = np.logical_and(pred_c, ~gt_c).sum()
            fn = np.logical_and(~pred_c, gt_c).sum()

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            inter = tp
            union = np.logical_or(pred_c, gt_c).sum()
            iou = inter / (union + 1e-10)

            precision_class[c] = precision
            recall_class[c] = recall
            f1_class[c] = f1
            iou_class[c] = iou

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels, preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[int(t), int(p)] += 1

        return {
            'm_iou': float(np.mean(iou_class)),
            'm_precision': float(np.mean(precision_class)),
            'm_recall': float(np.mean(recall_class)),
            'm_f1': float(np.mean(f1_class)),
            'iou_class': iou_class,
            'precision_class': precision_class,
            'recall_class': recall_class,
            'f1_class': f1_class,
            'confusion_matrix': cm,
            'class_support': cm.sum(axis=1),
        }

    def train_and_evaluate(self) -> Dict[str, object]:
        """Full training and evaluation loop."""
        print(f"\nTraining {len(self.classifiers)} classifiers for {self.epochs} epochs...")
        print(f"Learning rates: {sorted(self.learning_rates)[:5]}... ({len(self.learning_rates)} total)")

        for epoch in range(self.epochs):
            self.train_epoch(epoch)

        results = self.evaluate()
        return results


# ============================================================================
# Tool Implementation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming linear probing evaluation (memory efficient)"
    )
    parser.add_argument("--config-file", required=True, help="Sonata config file")
    parser.add_argument("--weight", required=True, help="Model checkpoint")
    parser.add_argument("--train-list", required=True, help="Training file list")
    parser.add_argument("--test-list", required=True, help="Test file list")
    parser.add_argument("--output-dir", default="linear_probe_streaming_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=48, help="Events per batch")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--network", type=str, default="teacher", choices=["student", "teacher"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr-list", type=str, default=None,
                       help="Comma-separated learning rates (default: grid search)")
    return parser.parse_args()


def load_model(config_path, weight_path, device):
    """Load Sonata model from config and checkpoint."""
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    print(f"Building model: {cfg.model['type']}")
    model = build_model(cfg.model)
    model = model.to(device)
    model.eval()

    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded successfully (epoch: {checkpoint.get('epoch', 'unknown')})")

    return model, cfg


def build_inference_dataset(cfg, file_list_path):
    """Build dataset for inference."""
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(files)} files in {file_list_path}")

    train_cfg = cfg.data.train.copy()
    train_cfg['data_list_file'] = file_list_path
    train_cfg['test_mode'] = False
    train_cfg['loop'] = 1

    grid_size = cfg.get('grid_size', 0.25)

    in_channels = cfg.model.get('backbone', {}).get('in_channels', 6)
    feat_keys = ("strength",) if in_channels == 3 else ("strength", "color")

    max_points_spherecrop=20480
    min_points_spherecrop=4096
    biased_spherecrop_radius=20.0

    transform = [
        #dict(type="SphereCrop", point_max=98304, mode="random"),
        dict(type="BiasedSphereCrop", 
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=biased_spherecrop_radius,
            point_max=max_points_spherecrop, 
            point_min=min_points_spherecrop, 
            prob_random=0.25,
            max_retries=100,
            fallback_to_random=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(type="Collect", keys=("coord", "grid_coord", "segment", "name", "grid_size"),
             offset_keys_dict=dict(offset="coord"), feat_keys=feat_keys),
    ]

    train_cfg['transform'] = transform
    dataset = build_dataset(train_cfg)
    return dataset, grid_size


def collate_fn(batch):
    """Custom collate function for point cloud data."""
    result = {}
    for key in batch[0].keys():
        if key == "offset":
            offsets = [0]
            for item in batch:
                offsets.append(offsets[-1] + item["offset"][-1].item())
            result[key] = torch.tensor(offsets[1:], dtype=torch.long)
        elif key == "name":
            result[key] = [item[key] for item in batch]
        elif key == "grid_size":
            result[key] = batch[0][key]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.cat([item[key] for item in batch], dim=0)
        else:
            result[key] = [item[key] for item in batch]
    return result


def get_feature_dim(model, sample_data, use_teacher, device):
    """Get feature dimension by running one forward pass."""
    for key in list(sample_data.keys()):
        if isinstance(sample_data[key], torch.Tensor):
            sample_data[key] = sample_data[key].to(device)
        elif key == 'grid_size':
            sample_data[key] = torch.tensor(sample_data[key], device=device)

    with torch.no_grad():
        network = model.teacher if use_teacher else model.student
        backbone = network['backbone']
        point = backbone(sample_data)

        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        return point.feat.shape[1]


def print_results(results, class_names, num_classes):
    """Print formatted results."""
    metrics = results['best_metrics']

    print(f"\nBest classifier: {results['best_classifier']}")
    lr_str = results['best_classifier'].replace('lr_', '').replace('_', '.')
    print(f"Best learning rate: {lr_str}")

    print(f"\nOverall metrics:")
    print(f"  mIoU:       {metrics['m_iou'] * 100:.2f}%")
    print(f"  mPrecision: {metrics['m_precision'] * 100:.2f}%")
    print(f"  mRecall:    {metrics['m_recall'] * 100:.2f}%")
    print(f"  mF1:        {metrics['m_f1'] * 100:.2f}%")

    # Per-class table
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Idx", justify="right")
        table.add_column("Class")
        table.add_column("IoU", justify="right")
        table.add_column("Prec", justify="right")
        table.add_column("Rec", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Support", justify="right")

        for i in range(num_classes):
            name = class_names[i] if i < len(class_names) else f"c{i}"
            table.add_row(
                str(i), str(name),
                f"{metrics['iou_class'][i]:.4f}",
                f"{metrics['precision_class'][i]:.4f}",
                f"{metrics['recall_class'][i]:.4f}",
                f"{metrics['f1_class'][i]:.4f}",
                str(int(metrics['class_support'][i])),
            )
        Console().print(table)
    except ImportError:
        print("\nPer-class metrics:")
        for i in range(num_classes):
            name = class_names[i] if i < len(class_names) else f"c{i}"
            print(f"  {i}: {name:>12} IoU={metrics['iou_class'][i]:.4f} F1={metrics['f1_class'][i]:.4f}")

    # Grid search results
    print("\nGrid search results (top 5 by mF1):")
    sorted_results = sorted(results['all_results'].items(), key=lambda x: -x[1]['m_f1'])
    for name, m in sorted_results[:5]:
        lr = name.replace('lr_', '').replace('_', '.')
        print(f"  LR {lr}: mF1={m['m_f1']*100:.2f}%")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, cfg = load_model(args.config_file, args.weight, args.device)
    use_teacher = args.network == "teacher"
    print(f"Using {'teacher' if use_teacher else 'student'} backbone")

    # Build datasets
    print("\nBuilding training dataset...")
    train_dataset, grid_size = build_inference_dataset(cfg, args.train_list)

    print("Building test dataset...")
    test_dataset, _ = build_inference_dataset(cfg, args.test_list)

    # Get class info
    class_names = cfg.data.get('names', [f'class_{i}' for i in range(cfg.data.num_classes)])
    num_classes = cfg.data.num_classes

    # Apply sample limits
    from torch.utils.data import Subset
    if args.max_train_samples:
        indices = list(range(min(len(train_dataset), args.max_train_samples)))
        train_dataset = Subset(train_dataset, indices)
    if args.max_test_samples:
        indices = list(range(min(len(test_dataset), args.max_test_samples)))
        test_dataset = Subset(test_dataset, indices)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # Get feature dimension
    print("Determining feature dimension...")
    sample = next(iter(train_loader))
    feature_dim = get_feature_dim(model, sample.copy(), use_teacher, args.device)
    print(f"Feature dimension: {feature_dim}")

    # Learning rates
    if args.lr_list:
        learning_rates = [float(x) for x in args.lr_list.split(',')]
    else:
        learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]

    # Create trainer
    trainer = StreamingLinearProbeTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        feature_dim=feature_dim,
        learning_rates=learning_rates,
        epochs=args.epochs,
        use_teacher=use_teacher,
        device=args.device,
    )

    # Train and evaluate
    print("\n" + "=" * 80)
    print("STREAMING LINEAR PROBE TRAINING")
    print("=" * 80)

    results = trainer.train_and_evaluate()

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print_results(results, class_names, num_classes)

    # Save results
    import json
    save_results = {
        'config_file': args.config_file,
        'weight_file': args.weight,
        'train_list': args.train_list,
        'test_list': args.test_list,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'feature_dim': feature_dim,
        'epochs': args.epochs,
        'best_classifier': results['best_classifier'],
        'best_mf1': float(results['best_mf1']),
        'm_iou': float(results['best_metrics']['m_iou']),
        'm_precision': float(results['best_metrics']['m_precision']),
        'm_recall': float(results['best_metrics']['m_recall']),
        'm_f1': float(results['best_metrics']['m_f1']),
        'iou_class': [float(x) for x in results['best_metrics']['iou_class']],
        'f1_class': [float(x) for x in results['best_metrics']['f1_class']],
        'class_support': [int(x) for x in results['best_metrics']['class_support']],
    }

    summary_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(summary_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to: {summary_path}")

    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")
    np.save(cm_path, results['best_metrics']['confusion_matrix'])
    print(f"Saved confusion matrix to: {cm_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
