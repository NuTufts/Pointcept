import numpy as np
import torch
import torch.distributed as dist

from .default import HookBase
from .builder import HOOKS

import pointcept.utils.comm as comm

@HOOKS.register_module()
class PretrainEvaluator(HookBase):
    def __init__(self, label="segment", write_cls_iou=True, every_n_steps=1, 
                max_train_events=250, max_test_events=250, 
                class_weights=None, class_names=None, prefix=""):
        self.write_cls_iou = write_cls_iou
        self.every_n_steps = every_n_steps
        self.max_train_events = max_train_events
        self.max_test_events = max_test_events
        self.prefix = prefix
        # support both single label and multiple labels
        self.labels = [label] if isinstance(label, str) else list(label)
        
        # support per-label class_weights
        # class_weights can be:
        # - None: no weights for any label
        # - list/array: same weights for all labels
        # - dict: {label_name: weights} for per-label weights
        if class_weights is None or not isinstance(class_weights, dict):
            # single set of weights or None - apply to all labels
            self.class_weights_dict = {label_name: class_weights for label_name in self.labels}
        else:
            # dict of per-label weights
            self.class_weights_dict = class_weights
        
        # support per-label class_names
        # class_names can be:
        # - None: use default names from cfg.data.names
        # - list: same names for all labels
        # - dict: {label_name: names_list} for per-label names
        if class_names is None or not isinstance(class_names, dict):
            # single set of names or None - apply to all labels
            self.class_names_dict = {label_name: class_names for label_name in self.labels}
        else:
            # dict of per-label names
            self.class_names_dict = class_names
        
    def after_step(self):
        if self.trainer.cfg.evaluate and self.every_n_steps > 0:
            # Calculate global iteration from epoch and iter within epoch
            # Note: epoch is stored as self.trainer.epoch, not in comm_info
            iter_per_epoch = len(self.trainer.train_loader)
            global_iter = self.trainer.comm_info['iter'] + iter_per_epoch * self.trainer.epoch
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.every_n_steps == 0:
            self.eval()

    def _unwrap_model(self):
        if isinstance(self.trainer.model, torch.nn.parallel.DistributedDataParallel):
            return self.trainer.model.module
        return self.trainer.model

    def get_backbone(self):
        model = self._unwrap_model()
        if hasattr(model, "teacher"): # sonata
            return model.teacher["backbone"]
        elif hasattr(model, "backbone"): # else
            return model.backbone
        else:
            raise ValueError(f"Model {model} has no backbone")
        
    def _process_batch_with_offsets(self, input_dict):
        """Process a batch and extract features properly using offsets to handle multiple events"""
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with torch.inference_mode():
            # Run backbone forward pass
            point = self.get_backbone()(input_dict)
            # Upsample features through pooling hierarchy
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent

        # Use voxelized features directly (no upsampling via inverse)
        # This is appropriate for linear probing and avoids issues when BiasedSphereCrop
        # invalidates the inverse indices from GridSample
        features = point.feat.cpu()  # [M, C] where M = number of voxels after crop

        # Use the point's offset (post-crop) for batch separation
        if hasattr(point, 'offset'):
            offsets = [0] + point.offset.cpu().tolist()
        else:
            offsets = [0] + input_dict['offset'].cpu().tolist()

        # Extract all label types (move to CPU immediately)
        all_labels = {}
        for label_name in self.labels:
            if not hasattr(point, label_name):
                self.trainer.logger.error(f"[PretrainEvaluator] Point has no '{label_name}' attribute")
                raise ValueError(f"Point must have '{label_name}' attribute")

            label_data = getattr(point, label_name).squeeze(-1).cpu()
            all_labels[label_name] = label_data  # [M]

        # Clear GPU memory
        del point
        torch.cuda.empty_cache()

        # Process features by batch using offsets
        batch_features = []
        batch_labels_dict = {label_name: [] for label_name in self.labels}

        # Use offsets to separate points from different events in the batch
        for i in range(len(offsets) - 1):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]

            # Extract features for this event (already on CPU)
            event_features = features[start_idx:end_idx]
            batch_features.append(event_features)
            # Extract labels for each label type
            for label_name in self.labels:
                event_labels = all_labels[label_name][start_idx:end_idx]
                batch_labels_dict[label_name].append(event_labels)

        return batch_features, batch_labels_dict

    def eval(self):
        # All ranks participate in evaluation
        self.trainer.model.eval()

        world_size = comm.get_world_size()
        rank = comm.get_rank()

        if rank == 0:
            self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.trainer.logger.info(f"[PretrainEvaluator] rank={rank}, world_size={world_size}")

        # Calculate per-rank event targets to ensure total events match max_train_events + max_test_events
        # Each rank collects its share of events
        per_rank_train_events = (self.max_train_events + world_size - 1) // world_size  # Ceiling division
        per_rank_test_events = (self.max_test_events + world_size - 1) // world_size  # Ceiling division
        per_rank_total_events = per_rank_train_events + per_rank_test_events

        self.trainer.logger.info(f"[PretrainEvaluator] Need {per_rank_total_events} events per rank "
                                 f"(train={per_rank_train_events}, test={per_rank_test_events})")

        # Collect features and labels from events (features shared, labels per label type)
        train_features = []
        train_labels_dict = {label_name: [] for label_name in self.labels}
        test_features = []
        test_labels_dict = {label_name: [] for label_name in self.labels}

        event_count = 0

        # All ranks iterate their shard of the distributed loader
        self.trainer.logger.info(f"[PretrainEvaluator] Starting to iterate val_loader (len={len(self.trainer.val_loader)})")
        for i, input_dict in enumerate(self.trainer.val_loader):
            if i % 10 == 0:
                self.trainer.logger.info(f"[PretrainEvaluator] Processing batch {i}, collected {event_count} events so far")
                # Periodically clear GPU cache
                torch.cuda.empty_cache()

            batch_features, batch_labels_dict = self._process_batch_with_offsets(input_dict)

            # Clear input dict to free memory
            del input_dict

            # Process each event in the batch
            for event_idx, event_features in enumerate(batch_features):
                if event_count < per_rank_train_events:
                    train_features.append(event_features)
                    for label_name in self.labels:
                        train_labels_dict[label_name].append(batch_labels_dict[label_name][event_idx])
                elif event_count < per_rank_total_events:
                    test_features.append(event_features)
                    for label_name in self.labels:
                        test_labels_dict[label_name].append(batch_labels_dict[label_name][event_idx])
                else:
                    break

                event_count += 1

            # Stop if we have enough events for this rank
            if event_count >= per_rank_total_events:
                self.trainer.logger.info(f"[PretrainEvaluator] Collected enough events ({event_count}), stopping iteration")
                break

        self.trainer.logger.info(f"[PretrainEvaluator] Done collecting. train={len(train_features)}, test={len(test_features)}")

        # Clear GPU cache before gathering/linear probing
        torch.cuda.empty_cache()

        # Gather all events from all ranks to rank 0
        if world_size > 1:
            self.trainer.logger.info(f"[PretrainEvaluator] Gathering from {world_size} ranks...")
            # Gather train features
            train_features_gathered = comm.gather(train_features, dst=0)
            self.trainer.logger.info(f"[PretrainEvaluator] Gathered train features")
            # Gather test features
            test_features_gathered = comm.gather(test_features, dst=0)
            self.trainer.logger.info(f"[PretrainEvaluator] Gathered test features")
            # Gather train labels for each label type
            train_labels_gathered_dict = {}
            test_labels_gathered_dict = {}
            for label_name in self.labels:
                train_labels_gathered_dict[label_name] = comm.gather(train_labels_dict[label_name], dst=0)
                test_labels_gathered_dict[label_name] = comm.gather(test_labels_dict[label_name], dst=0)
            self.trainer.logger.info(f"[PretrainEvaluator] Gathered all labels")

            if rank == 0:
                # Flatten gathered lists and truncate to exact target counts
                train_features = [f for features_list in train_features_gathered for f in features_list][:self.max_train_events]
                test_features = [f for features_list in test_features_gathered for f in features_list][:self.max_test_events]

                for label_name in self.labels:
                    train_labels_dict[label_name] = [l for labels_list in train_labels_gathered_dict[label_name] for l in labels_list][:self.max_train_events]
                    test_labels_dict[label_name] = [l for labels_list in test_labels_gathered_dict[label_name] for l in labels_list][:self.max_test_events]
                self.trainer.logger.info(f"[PretrainEvaluator] Flattened gathered data on rank 0")
            else:
                # Non-rank-0 processes set model back to train and wait
                self.trainer.logger.info(f"[PretrainEvaluator] Rank {rank} waiting at synchronize...")
                self.trainer.model.train()
                comm.synchronize()
                self.trainer.logger.info(f"[PretrainEvaluator] Rank {rank} done with synchronize")
                return
        else:
            # Single process case - truncate to exact counts
            self.trainer.logger.info(f"[PretrainEvaluator] Single process mode, truncating to exact counts")
            train_features = train_features[:self.max_train_events]
            test_features = test_features[:self.max_test_events]
            for label_name in self.labels:
                train_labels_dict[label_name] = train_labels_dict[label_name][:self.max_train_events]
                test_labels_dict[label_name] = test_labels_dict[label_name][:self.max_test_events]
        
        # Only rank 0 reaches here (or single-process case)
        # Concatenate features (shared across all labels)
        if not train_features or not test_features:
            self.trainer.logger.error("Not enough events for train/test split")
            self.trainer.model.train()
            if world_size > 1:
                comm.synchronize()
            return
            
        X_train = torch.cat(train_features, dim=0)
        X_test = torch.cat(test_features, dim=0)

        self.trainer.logger.info(f"Train events: {len(train_features)}, Test events: {len(test_features)}")
        self.trainer.logger.info(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
        
        # Now evaluate for each label type
        for label_name in self.labels:
            self.trainer.logger.info(f"\n{'='*60}\nEvaluating label: {label_name}\n{'='*60}")
            
            # Concatenate labels for this label type
            y_train = torch.cat(train_labels_dict[label_name], dim=0)
            y_test = torch.cat(test_labels_dict[label_name], dim=0)
            
            # Determine prefix for logging
            if len(self.labels) > 1:
                # multiple labels: use label_name as prefix
                eval_prefix = label_name if not self.prefix else f"{self.prefix}_{label_name}"
            else:
                # single label: use provided prefix or default
                eval_prefix = self.prefix if self.prefix else label_name
            
            # Get class_weights for this label
            label_class_weights = self.class_weights_dict.get(label_name, None)
            
            # Get class_names for this label (None means use default from cfg)
            label_class_names = self.class_names_dict.get(label_name, None)
            
            # Run evaluation for this label
            self._evaluate_single_label(X_train, y_train, X_test, y_test, eval_prefix, label_class_weights, label_class_names)
        
        # Set model back to train mode
        self.trainer.model.train()
        
        # Synchronize before returning (ensure all ranks finish together)
        if world_size > 1:
            comm.synchronize()
    
    def _evaluate_single_label(self, X_train, y_train, X_test, y_test, eval_prefix, label_class_weights, label_class_names):
        """Train and evaluate a grid of linear classifiers for a single label type."""
        self.trainer.logger.info(f"[PretrainEvaluator] Starting linear probing for {eval_prefix}")
        self.trainer.logger.info(f"[PretrainEvaluator] X_train: {X_train.shape}, y_train: {y_train.shape}")
        self.trainer.logger.info(f"[PretrainEvaluator] X_test: {X_test.shape}, y_test: {y_test.shape}")

        from pointcept.engines.hooks.eval.linear import LinearProbingTrainer, LinearProbingConfig

        # Use provided class names or fall back to default
        if label_class_names is None:
            label_class_names = self.trainer.cfg.data.names

        num_classes = int(y_train.max().item()) + 1

        cfg = LinearProbingConfig()
        trainer = LinearProbingTrainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            logger=self.trainer.logger,
            config=cfg,
        )
        metrics = trainer.train_and_evaluate()

        m_iou = metrics["m_iou"]
        m_precision = metrics["m_precision"]
        m_recall = metrics["m_recall"]
        m_f1 = metrics["m_f1"]
        iou_class = metrics["iou_class"]
        precision_class = metrics["precision_class"]
        recall_class = metrics["recall_class"]
        f1_class = metrics["f1_class"]
        cm = metrics["confusion_matrix"]
        class_support = metrics["class_support"]

        self.trainer.storage.put_scalar(f"{eval_prefix}_val_intersection", iou_class * (class_support + 1e-10))
        self.trainer.storage.put_scalar(f"{eval_prefix}_val_union", (class_support + 1e-10))
        self.trainer.storage.put_scalar(f"{eval_prefix}_val_target", class_support)

        self.trainer.logger.info(
            "Val result: mIoU/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_precision, m_recall, m_f1
            )
        )

        from rich.table import Table
        from rich.console import Console

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ClassIdx", justify="right")
        table.add_column("Name")
        table.add_column("IoU", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Support", justify="right")

        for i in range(num_classes):
            table.add_row(
                str(i),
                str(label_class_names[i]),
                f"{iou_class[i]:.4f}",
                f"{precision_class[i]:.4f}",
                f"{recall_class[i]:.4f}",
                f"{f1_class[i]:.4f}",
                str(class_support[i]),
            )

        console = Console(file=None, width=100, record=True)
        console.print(table)
        table_str = console.export_text()  # noqa: F841

        import pandas as pd
        label_names = [str(n) for n in label_class_names[:num_classes]]
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        self.trainer.logger.info("Confusion Matrix (rows=true, cols=pred):\n" + cm_df.to_string())

        _prefix = eval_prefix
        eval_prefix = eval_prefix + "/"
        if eval_prefix == "segment/":
            eval_prefix = ""

        if self.trainer.writer is not None:
            # pass to wandb
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mIoU", m_iou, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mPrecision", m_precision, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mRecall", m_recall, self.trainer.writer.run.step)
            self.trainer.writer.add_scalar(f"{eval_prefix}val/mF1", m_f1, self.trainer.writer.run.step)

            if self.write_cls_iou:
                for i in range(num_classes):
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} IoU",
                        iou_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} F1",
                        f1_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Precision",
                        precision_class[i],
                        self.trainer.writer.run.step
                    )
                    self.trainer.writer.add_scalar(
                        f"{eval_prefix}val/cls_{i}-{label_class_names[i]} Recall",
                        recall_class[i],
                        self.trainer.writer.run.step
                    )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        if "current_metric_value" not in self.trainer.comm_info.keys():
            self.trainer.comm_info["current_metric_name"] = "mF1"
        self.trainer.comm_info["current_metric_value"] = m_f1
        self.trainer.comm_info[f"{_prefix}_current_metric_value"] = m_f1
