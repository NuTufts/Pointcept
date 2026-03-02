"""
Shower Origin Prediction Model

Query-conditioned architecture for identifying the origin points of
electromagnetic showers in LArTPC data. Uses a frozen Sonata encoder
with Slot Attention for query formation and multi-round cross-attention.

Author: Claude Code Assistant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .slot_attention import SlotAttention
from .virtual_grid import VirtualGridGenerator


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block where queries attend to keys/values.

    Implements standard transformer cross-attention with:
    - Multi-head attention
    - Pre-normalization
    - MLP with residual connection

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension multiplier
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Multi-head attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, return_attn=False):
        """
        Cross-attention forward pass.

        Args:
            queries: (B, Q, D) or (Q, D) query vectors
            keys: (B, K, D) or (K, D) key vectors
            values: (B, K, D) or (K, D) value vectors
            return_attn: Whether to return attention weights

        Returns:
            output: Updated query vectors
            attn_weights: (optional) Attention weights (B, num_heads, Q, K)
        """
        # Handle unbatched inputs
        squeeze_batch = False
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
            squeeze_batch = True

        B, Q, D = queries.shape
        _, K, _ = keys.shape

        # Pre-normalize
        q = self.norm1(queries)
        kv = self.norm_kv(keys)  # Assume keys == values or both normalized same way

        # Project to Q, K, V
        q = self.q_proj(q).reshape(B, Q, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = self.k_proj(kv).reshape(B, K, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = self.v_proj(values if values is not kv else kv).reshape(
            B, K, self.num_heads, D // self.num_heads
        ).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(-50, 50)
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, Q, D)
        out = self.out_proj(out)

        # Residual connection
        queries = queries + out

        # MLP with residual
        queries = queries + self.mlp(self.norm2(queries))

        if squeeze_batch:
            queries = queries.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        if return_attn:
            return queries, attn_weights
        return queries


class OriginScoreHead(nn.Module):
    """
    Head for predicting per-point origin scores and distances.

    Takes event features conditioned on shower query and produces:
    - Origin score: Probability each point is the origin (0-1)
    - Distance: Predicted distance to origin (optional)

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        predict_distance: Whether to predict distance in addition to score
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        predict_distance: bool = True,
    ):
        super().__init__()
        self.predict_distance = predict_distance

        # Shared MLP layers
        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # Score head (sigmoid output for probability)
        self.score_head = nn.Linear(hidden_channels, 1)

        # Distance head (positive output via softplus)
        if predict_distance:
            self.distance_head = nn.Linear(hidden_channels, 1)

    def forward(self, features):
        """
        Args:
            features: (N, D) or (B, N, D) point features

        Returns:
            scores: (N,) or (B, N) origin probability scores
            distances: (N,) or (B, N) predicted distances (if predict_distance=True)
        """
        squeeze_batch = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze_batch = True

        x = self.shared(features)

        # Origin score: return both logits and sigmoid scores
        # Cast to float32 to avoid NaN from bfloat16 overflow
        score_logits = self.score_head(x).float().squeeze(-1)
        scores = torch.sigmoid(score_logits.clamp(-50, 50))

        output = {'scores': scores, 'score_logits': score_logits}

        if self.predict_distance:
            # Distance (positive via softplus, cast to float32 for numerical safety)
            distances = F.softplus(self.distance_head(x).float().clamp(-50, 50)).squeeze(-1)
            output['distances'] = distances

        if squeeze_batch:
            output = {k: v.squeeze(0) for k, v in output.items()}

        return output


@MODELS.register_module("ShowerOriginPredictor")
class ShowerOriginPredictor(nn.Module):
    """
    Query-conditioned model for shower origin prediction.

    Architecture:
    1. Frozen Sonata encoder extracts features for all event points
    2. Slot Attention aggregates shower cluster features into K query slots
    3. Multi-round cross-attention: slots query the full event
    4. Per-point score head predicts origin likelihood

    Args:
        backbone: Config dict for Sonata backbone
        backbone_out_channels: Output channels from backbone after upcast
        num_slots: Number of slot attention slots (default: 4)
        slot_iterations: Slot attention iterations (default: 3)
        num_cross_attn_layers: Number of cross-attention refinement layers (default: 3)
        num_heads: Attention heads in cross-attention (default: 8)
        hidden_channels: Hidden dimension for score head
        predict_distance: Whether to predict distance in addition to score
        criteria: Loss function config
        freeze_backbone: Whether to freeze backbone (default: True)
        gaussian_sigma: Sigma for Gaussian soft labels (in coordinate units)
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels: int,
        num_slots: int = 4,
        slot_iterations: int = 3,
        num_cross_attn_layers: int = 3,
        num_heads: int = 8,
        hidden_channels: int = 256,
        predict_distance: bool = True,
        criteria=None,
        freeze_backbone: bool = True,
        gaussian_sigma: float = 4.0,
    ):
        super().__init__()

        self.backbone_out_channels = backbone_out_channels
        self.num_slots = num_slots
        self.freeze_backbone = freeze_backbone
        self.gaussian_sigma = gaussian_sigma
        self.predict_distance = predict_distance

        # Build backbone (Sonata model)
        self.backbone = build_model(backbone)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Slot Attention for query formation from shower
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=backbone_out_channels,
            num_iterations=slot_iterations,
        )

        # Multi-round cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=backbone_out_channels,
                num_heads=num_heads,
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Combine slot information with point features
        # Concatenate: [point_features, aggregated_query_features]
        self.combiner = nn.Sequential(
            nn.Linear(backbone_out_channels * 2, backbone_out_channels),
            nn.LayerNorm(backbone_out_channels),
            nn.GELU(),
        )

        # Per-point scoring head
        self.score_head = OriginScoreHead(
            in_channels=backbone_out_channels,
            hidden_channels=hidden_channels,
            predict_distance=predict_distance,
        )

        # Loss functions
        self.criteria = build_criteria(criteria) if criteria else None

    def encode_event(self, input_dict):
        """
        Encode full event using frozen backbone.

        Args:
            input_dict: Dict with coord, feat, offset, etc.

        Returns:
            point: Point object with features
            feat: (N, D) per-point features
        """
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(input_dict, return_point=True)
        else:
            result = self.backbone(input_dict, return_point=True)

        point = result["point"]
        return point, point.feat

    def forward(self, input_dict, return_attention=False):
        """
        Forward pass for shower origin prediction.

        Args:
            input_dict: Dict containing:
                - coord: (N, 3) coordinates
                - feat: (N, C) input features
                - offset: (B,) batch offsets
                - shower_mask: (N,) boolean mask for shower points
                - origin_coord: (B, 3) true origin coordinates (training only)
                - origin_distance: (N,) distance to origin (training only)

        Returns:
            Dict with:
                - origin_scores: (N,) per-point origin scores
                - origin_distances: (N,) predicted distances (if predict_distance)
                - loss: scalar loss (if training)
                - attention_weights: list of attention weights (if return_attention)
        """
        # Encode full event
        point, event_features = self.encode_event(input_dict)

        # Get shower mask
        shower_mask = input_dict["shower_mask"]

        # Extract shower point features
        shower_features = event_features[shower_mask]  # (S, D)

        # Form slot queries from shower via Slot Attention
        # Add batch dimension for slot attention
        shower_slots, slot_attn = self.slot_attention(
            shower_features.unsqueeze(0)
        )
        shower_slots = shower_slots.squeeze(0)  # (K, D)

        # Multi-round cross-attention: slots query the full event
        queries = shower_slots
        all_attentions = []

        for layer in self.cross_attn_layers:
            if return_attention:
                queries, attn = layer(
                    queries=queries,
                    keys=event_features,
                    values=event_features,
                    return_attn=True,
                )
                all_attentions.append(attn)
            else:
                queries = layer(
                    queries=queries,
                    keys=event_features,
                    values=event_features,
                )

        # Aggregate slot information
        # Mean pool the refined slot queries
        query_summary = queries.mean(dim=0)  # (D,)

        # Combine with point features
        # Broadcast query to all points and concatenate
        query_broadcast = query_summary.unsqueeze(0).expand(event_features.size(0), -1)
        combined = torch.cat([event_features, query_broadcast], dim=-1)
        combined = self.combiner(combined)

        # Predict scores and distances
        predictions = self.score_head(combined)

        result_dict = {
            "origin_scores": predictions["scores"],
        }

        if self.predict_distance:
            result_dict["origin_distances"] = predictions["distances"]

        if return_attention:
            result_dict["attention_weights"] = all_attentions
            result_dict["slot_attention"] = slot_attn

        # Compute loss if training
        if self.training and "origin_coord" in input_dict:
            loss = self.compute_loss(
                predictions=predictions,
                point=point,
                input_dict=input_dict,
            )
            result_dict["loss"] = loss

        # Evaluation mode with ground truth
        elif "origin_coord" in input_dict:
            loss = self.compute_loss(
                predictions=predictions,
                point=point,
                input_dict=input_dict,
            )
            result_dict["loss"] = loss

        return result_dict

    def compute_loss(self, predictions, point, input_dict):
        """
        Compute combined loss for origin prediction.

        Uses:
        - Gaussian soft labels for score loss (handles class imbalance)
        - L1 loss for distance regression

        Args:
            predictions: Dict with 'scores' and optionally 'distances'
            point: Point object with coordinates
            input_dict: Dict with ground truth

        Returns:
            loss: Scalar loss value
        """
        losses = {}

        # Get predictions
        pred_scores = predictions["scores"]

        # Compute distances from each point to origin
        origin_coord = input_dict["origin_coord"]  # (B, 3) or (3,)

        # Handle batched vs unbatched
        if origin_coord.dim() == 1:
            origin_coord = origin_coord.unsqueeze(0)

        # Get point coordinates
        if hasattr(point, 'coord'):
            coords = point.coord  # (N, 3)
        else:
            coords = input_dict["coord"]

        # For now, assume single sample (no batch dimension in points)
        # TODO: Handle batched data properly with offsets
        origin = origin_coord[0]  # (3,)
        distances_to_origin = torch.norm(coords - origin.unsqueeze(0), dim=-1)  # (N,)

        # Gaussian soft labels: higher score for points closer to origin
        target_scores = torch.exp(
            -0.5 * (distances_to_origin / self.gaussian_sigma) ** 2
        )

        # Score loss: BCE with logits for numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            if "score_logits" in predictions:
                score_loss = F.binary_cross_entropy_with_logits(
                    predictions["score_logits"].float(),
                    target_scores.float(),
                    reduction='mean'
                )
            else:
                score_loss = F.binary_cross_entropy(
                    pred_scores.float().clamp(1e-7, 1 - 1e-7),
                    target_scores.float(),
                    reduction='mean'
                )
        losses["score_loss"] = score_loss

        # Distance loss
        if self.predict_distance and "distances" in predictions:
            pred_distances = predictions["distances"]
            distance_loss = F.smooth_l1_loss(
                pred_distances,
                distances_to_origin,
                reduction='mean'
            )
            losses["distance_loss"] = distance_loss

        # Combine losses
        total_loss = losses["score_loss"]
        if "distance_loss" in losses:
            total_loss = total_loss + 0.1 * losses["distance_loss"]

        return total_loss


@MODELS.register_module("ShowerOriginPredictorV2")
class ShowerOriginPredictorV2(nn.Module):
    """
    Alternative version that works with DefaultSegmentorV2-style interface.

    This version expects the shower mask to be provided differently and
    is designed to integrate more easily with existing Pointcept training
    infrastructure.

    Instead of a single forward pass, this model:
    1. Encodes the full event
    2. Uses shower_mask from the data dict to identify shower points
    3. Predicts origin scores for all points

    The main difference from V1 is the data interface and loss computation.
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels: int,
        num_slots: int = 4,
        slot_iterations: int = 3,
        num_cross_attn_layers: int = 3,
        num_heads: int = 8,
        hidden_channels: int = 256,
        predict_distance: bool = True,
        criteria=None,
        freeze_backbone: bool = True,
        gaussian_sigma: float = 4.0,
        score_loss_weight: float = 1.0,
        distance_loss_weight: float = 0.1,
    ):
        super().__init__()

        self.backbone_out_channels = backbone_out_channels
        self.num_slots = num_slots
        self.freeze_backbone = freeze_backbone
        self.gaussian_sigma = gaussian_sigma
        self.predict_distance = predict_distance
        self.score_loss_weight = score_loss_weight
        self.distance_loss_weight = distance_loss_weight

        # Build backbone (Sonata model)
        self.backbone = build_model(backbone)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Slot Attention for query formation from shower
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=backbone_out_channels,
            num_iterations=slot_iterations,
        )

        # Multi-round cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=backbone_out_channels,
                num_heads=num_heads,
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Combine slot information with point features
        self.combiner = nn.Sequential(
            nn.Linear(backbone_out_channels * 2, backbone_out_channels),
            nn.LayerNorm(backbone_out_channels),
            nn.GELU(),
        )

        # Per-point scoring head
        self.score_head = OriginScoreHead(
            in_channels=backbone_out_channels,
            hidden_channels=hidden_channels,
            predict_distance=predict_distance,
        )

        # Optional: custom criteria if provided
        self.criteria = build_criteria(criteria) if criteria else None

    def forward(self, input_dict):
        """
        Forward pass compatible with Pointcept training.

        Args:
            input_dict: Dict with standard Pointcept keys plus:
                - shower_mask: (N,) boolean mask for shower points
                - origin_distance: (N,) ground truth distance to origin (training)

        Returns:
            Dict with loss (training) or predictions (inference)
        """
        # Encode full event using backbone
        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(input_dict, return_point=True)
        else:
            result = self.backbone(input_dict, return_point=True)

        point = result["point"]
        event_features = point.feat  # (N, D)

        # Get shower mask
        shower_mask = input_dict["shower_mask"].bool()

        # Extract shower point features
        shower_features = event_features[shower_mask]  # (S, D)

        # Handle case where shower has very few points
        if shower_features.size(0) < self.num_slots:
            # Pad with mean feature
            mean_feat = shower_features.mean(dim=0, keepdim=True)
            padding = mean_feat.expand(self.num_slots - shower_features.size(0), -1)
            shower_features = torch.cat([shower_features, padding], dim=0)

        # Slot Attention: aggregate shower into K slots
        shower_slots, _ = self.slot_attention(shower_features.unsqueeze(0))
        shower_slots = shower_slots.squeeze(0)  # (K, D)

        # Multi-round cross-attention
        queries = shower_slots
        for layer in self.cross_attn_layers:
            queries = layer(
                queries=queries,
                keys=event_features,
                values=event_features,
            )

        # Aggregate and combine
        query_summary = queries.mean(dim=0)
        query_broadcast = query_summary.unsqueeze(0).expand(event_features.size(0), -1)
        combined = torch.cat([event_features, query_broadcast], dim=-1)
        combined = self.combiner(combined)

        # Predict
        predictions = self.score_head(combined)

        return_dict = {}

        # Training / evaluation with ground truth
        if self.training or "origin_distance" in input_dict:
            # Ground truth distances
            gt_distances = input_dict["origin_distance"]

            # Gaussian soft labels from distances
            target_scores = torch.exp(
                -0.5 * (gt_distances / self.gaussian_sigma) ** 2
            )

            # Score loss: BCE with logits for numerical stability
            with torch.amp.autocast('cuda', enabled=False):
                if "score_logits" in predictions:
                    score_loss = F.binary_cross_entropy_with_logits(
                        predictions["score_logits"].float(),
                        target_scores.float(),
                        reduction='mean'
                    )
                else:
                    score_loss = F.binary_cross_entropy(
                        predictions["scores"].float().clamp(1e-7, 1 - 1e-7),
                        target_scores.float(),
                        reduction='mean'
                    )

            loss = self.score_loss_weight * score_loss

            # Distance loss
            if self.predict_distance:
                distance_loss = F.smooth_l1_loss(
                    predictions["distances"],
                    gt_distances,
                    reduction='mean'
                )
                loss = loss + self.distance_loss_weight * distance_loss
                return_dict["distance_loss"] = distance_loss

            return_dict["loss"] = loss
            return_dict["score_loss"] = score_loss

        # Add predictions to output
        if not self.training:
            return_dict["origin_scores"] = predictions["scores"]
            if self.predict_distance:
                return_dict["origin_distances"] = predictions["distances"]

        return return_dict


class OriginClassificationHead(nn.Module):
    """
    Classification head for origin type: INSIDE (0), OUTSIDE (1), ON_TRACK (2).

    For each slot, predicts the origin type of the corresponding shower.

    Args:
        dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 128)
        num_classes: Number of origin classes (default: 3)
    """

    def __init__(self, dim, hidden_dim=128, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, slot_feature):
        """
        Args:
            slot_feature: (D,) single slot feature vector

        Returns:
            logits: (num_classes,) class logits
        """
        return self.mlp(slot_feature)


@MODELS.register_module("ShowerOriginPredictorV3")
class ShowerOriginPredictorV3(nn.Module):
    """
    Multi-shower origin prediction with virtual grid and Hungarian matching.

    Key differences from V2:
    1. Virtual grid points for empty-space origin prediction
    2. Per-slot predictions (each slot predicts one shower's origin)
    3. Hungarian matching for slot-to-shower assignment during training
    4. Per-shower inside/outside classification head

    Architecture:
    1. Frozen Sonata encoder extracts features for all event points
    2. Shower features extracted via union of all shower masks
    3. Slot Attention aggregates shower features into K query slots
    4. (Optional) Virtual grid points generated and features interpolated
    5. Cross-attention: slots query all points (real + virtual)
    6. Per-slot: broadcast slot to points, combine, predict score/distance/class
    7. Hungarian matching assigns slots to ground truth showers for loss

    Args:
        backbone: Config dict for Sonata backbone
        backbone_out_channels: Output channels from backbone after upcast (496)
        num_slots: Number of slot attention slots (default: 4)
        slot_iterations: Slot attention iterations (default: 3)
        num_cross_attn_layers: Number of cross-attention layers (default: 3)
        num_heads: Attention heads in cross-attention (default: 8)
        hidden_channels: Hidden dimension for score head (default: 256)
        predict_distance: Whether to predict distance (default: True)
        criteria: Loss function config (optional)
        freeze_backbone: Whether to freeze backbone weights (default: True)
        gaussian_sigma: Sigma for Gaussian soft labels in cm (default: 4.0)
        score_loss_weight: Weight for score loss (default: 1.0)
        distance_loss_weight: Weight for distance loss (default: 0.1)
        classification_loss_weight: Weight for inside/outside loss (default: 1.0)
        no_object_loss_weight: Penalty weight for unmatched slots (default: 0.1)
        virtual_grid_enabled: Whether to use virtual grid (default: True)
        dense_spacing: Dense grid spacing in cm (default: 2.0)
        dense_radius: Dense grid radius around shower centroid in cm (default: 30.0)
        sparse_spacing: Sparse grid spacing in cm (default: 10.0)
        k_neighbors: k for kNN interpolation (default: 8)
        pos_embed_dim: Positional encoding dimension (default: 64)
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels: int,
        num_slots: int = 4,
        slot_iterations: int = 3,
        num_cross_attn_layers: int = 3,
        num_heads: int = 8,
        hidden_channels: int = 256,
        predict_distance: bool = True,
        criteria=None,
        freeze_backbone: bool = True,
        gaussian_sigma: float = 4.0,
        score_loss_weight: float = 1.0,
        distance_loss_weight: float = 0.1,
        classification_loss_weight: float = 1.0,
        # Origin classification
        num_origin_classes: int = 3,
        # Virtual grid params
        virtual_grid_enabled: bool = True,
        dense_spacing: float = 2.0,
        dense_radius: float = 30.0,
        sparse_spacing: float = 10.0,
        tpc_bounds: dict = None,
        k_neighbors: int = 8,
        pos_embed_dim: int = 64,
    ):
        super().__init__()

        # Store config
        self.backbone_out_channels = backbone_out_channels
        self.num_slots = num_slots
        self.freeze_backbone = freeze_backbone
        self.gaussian_sigma = gaussian_sigma
        self.predict_distance = predict_distance
        self.score_loss_weight = score_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.virtual_grid_enabled = virtual_grid_enabled

        # Build backbone (Sonata model)
        self.backbone = build_model(backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Free unused Sonata components to save GPU memory.
            # When return_point=True, only teacher.backbone is used.
            if hasattr(self.backbone, 'student'):
                del self.backbone.student
            if hasattr(self.backbone, 'teacher'):
                for key in list(self.backbone.teacher.keys()):
                    if key != 'backbone':
                        del self.backbone.teacher[key]

        # Slot Attention for query formation from shower features
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=backbone_out_channels,
            num_iterations=slot_iterations,
        )

        # Multi-round cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=backbone_out_channels,
                num_heads=num_heads,
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Final LayerNorm after cross-attention stack to prevent
        # residual connections from accumulating extreme values
        self.query_norm = nn.LayerNorm(backbone_out_channels)

        # Virtual grid generator (NEW in V3)
        self.virtual_grid = None
        if virtual_grid_enabled:
            self.virtual_grid = VirtualGridGenerator(
                backbone_dim=backbone_out_channels,
                dense_spacing=dense_spacing,
                dense_radius=dense_radius,
                sparse_spacing=sparse_spacing,
                tpc_bounds=tpc_bounds,
                k_neighbors=k_neighbors,
                pos_embed_dim=pos_embed_dim,
            )

        # Per-slot combiner: concatenate [point_features, slot_broadcast] -> backbone_dim
        self.combiner = nn.Sequential(
            nn.Linear(backbone_out_channels * 2, backbone_out_channels),
            nn.LayerNorm(backbone_out_channels),
            nn.GELU(),
        )

        # Per-point score head (same architecture as V2)
        self.score_head = OriginScoreHead(
            in_channels=backbone_out_channels,
            hidden_channels=hidden_channels,
            predict_distance=predict_distance,
        )

        # Classification head (V3): per-slot origin type (INSIDE/OUTSIDE/ON_TRACK)
        self.num_origin_classes = num_origin_classes
        self.classification_head = OriginClassificationHead(
            dim=backbone_out_channels,
            hidden_dim=hidden_channels,
            num_classes=num_origin_classes,
        )

        # Optional: custom criteria if provided
        self.criteria = build_criteria(criteria) if criteria else None

    def forward(self, input_dict):
        """
        Forward pass for multi-shower origin prediction.

        Args:
            input_dict: Dict with standard Pointcept keys plus:
                - shower_mask: (N,) boolean mask (union of all showers), OR
                  list of (N,) boolean masks, one per shower
                - origin_coords: (J, 3) ground truth origin coordinates for J showers
                - origin_types: (J,) ground truth types (0=inside, 1=outside TPC)
                - num_showers: int, number of showers in this event
                - origin_coord: (3,) or (B,3) single origin (V2-style fallback)
                - origin_distance: (N,) distance to origin (V2-style fallback)

        Returns:
            Dict with:
                - loss: scalar loss (training)
                - score_loss, distance_loss, cls_loss: component losses
                - per_slot_scores: list of (N,) score tensors (inference)
                - per_slot_classifications: list of (1,) logits (inference)
                - matched_indices: (row_ind, col_ind) from Hungarian (training)
        """
        # ==================================================================
        # 1. Encode full event with backbone
        # ==================================================================
        # PT-v3m2's GridPooling treats 'origin_coord' as per-point coords (N,3)
        # for pooling, but our dataset uses it for the shower origin point (3,).
        # Save and remove shower-specific keys before backbone forward.
        _saved_keys = {}
        for key in ("origin_coord", "origin_type", "origin_distance",
                     "shower_mask", "start_coord"):
            if key in input_dict:
                _saved_keys[key] = input_dict.pop(key)

        if self.freeze_backbone:
            with torch.no_grad():
                result = self.backbone(input_dict, return_point=True)
        else:
            result = self.backbone(input_dict, return_point=True)

        # Restore saved keys
        input_dict.update(_saved_keys)

        point = result["point"]
        event_features = point.feat  # (N_total_out, D) — concatenated across batch
        real_coords = point.coord  # (N_total_out, 3)

        # ==================================================================
        # 2. Determine batch size and per-sample boundaries
        # ==================================================================
        output_offset = getattr(point, 'offset', None)
        input_offset = input_dict.get("offset", None)

        if output_offset is not None:
            B = output_offset.shape[0]
        elif input_offset is not None:
            B = input_offset.shape[0]
        else:
            B = 1

        # Gather full-batch tensors
        shower_mask_all = input_dict.get("shower_mask", None)
        input_coords_all = input_dict.get("coord", None)
        origin_coord_all = input_dict.get("origin_coord", None)
        origin_type_all = input_dict.get("origin_type", None)

        # Reshape origin_coord from (B*3,) to (B, 3) when batched
        if origin_coord_all is not None:
            if origin_coord_all.dim() == 1 and B > 1:
                origin_coord_all = origin_coord_all.view(B, 3)
            elif origin_coord_all.dim() == 1 and B == 1:
                origin_coord_all = origin_coord_all.unsqueeze(0)

        # ==================================================================
        # 3-8. Process each sample in the batch independently
        # ==================================================================
        return_dict = {}
        batch_losses = []
        batch_eval_outputs = []

        for b in range(B):
            # --- Per-sample slicing ---
            if output_offset is not None:
                out_s = 0 if b == 0 else output_offset[b - 1].item()
                out_e = output_offset[b].item()
            else:
                out_s, out_e = 0, event_features.shape[0]

            if input_offset is not None:
                in_s = 0 if b == 0 else input_offset[b - 1].item()
                in_e = input_offset[b].item()
            else:
                in_s = 0
                in_e = (input_coords_all.shape[0]
                        if input_coords_all is not None else 0)

            sample_features = event_features[out_s:out_e]  # (N_out_b, D)
            sample_coords = real_coords[out_s:out_e]        # (N_out_b, 3)
            N_out_b = sample_features.shape[0]

            # --- Shower mask (input resolution → output resolution) ---
            sample_mask = None
            if shower_mask_all is not None:
                sample_mask = shower_mask_all[in_s:in_e]
                sample_input_coords = (
                    input_coords_all[in_s:in_e]
                    if input_coords_all is not None else None
                )
                if (sample_input_coords is not None
                        and sample_mask.shape[0] != N_out_b):
                    sample_mask = self._downsample_mask(
                        sample_mask, sample_input_coords, sample_coords
                    )

            if sample_mask is not None:
                shower_features = sample_features[sample_mask.bool()]
            else:
                shower_features = sample_features

            if shower_features.size(0) < self.num_slots:
                mean_feat = shower_features.mean(dim=0, keepdim=True)
                padding = mean_feat.expand(
                    self.num_slots - shower_features.size(0), -1
                )
                shower_features = torch.cat([shower_features, padding], dim=0)

            # --- Slot Attention (float32) ---
            with torch.amp.autocast('cuda', enabled=False):
                shower_slots, _ = self.slot_attention(
                    shower_features.float().unsqueeze(0)
                )
                shower_slots = shower_slots.squeeze(0)  # (K, D)

            # Debug: log tensor stats at key points
            if self.training:
                import logging as _logging
                _dbg = _logging.getLogger("nan_debug")
                _dbg.info(
                    f"[sample {b}] backbone_feats: "
                    f"abs_max={sample_features.abs().max().item():.4f}, "
                    f"shower_feats({shower_features.shape[0]} pts): "
                    f"abs_max={shower_features.abs().max().item():.4f}, "
                    f"slots: abs_max={shower_slots.abs().max().item():.4f}"
                )

            # --- Virtual grid (optional) ---
            if self.virtual_grid is not None:
                shower_centroids = self._compute_shower_centroids(
                    sample_coords, sample_mask
                )
                virtual_coords, virtual_features, _ = self.virtual_grid(
                    sample_coords, sample_features, shower_centroids
                )
                if virtual_coords.shape[0] > 0:
                    all_features = torch.cat(
                        [sample_features, virtual_features], dim=0
                    )
                    all_coords = torch.cat(
                        [sample_coords, virtual_coords], dim=0
                    )
                else:
                    all_features = sample_features
                    all_coords = sample_coords
            else:
                all_features = sample_features
                all_coords = sample_coords

            # --- Cross-attention (float32) ---
            with torch.amp.autocast('cuda', enabled=False):
                queries = shower_slots.float()
                all_features_f32 = all_features.float()
                for _layer_idx, layer in enumerate(self.cross_attn_layers):
                    queries = layer(
                        queries=queries,
                        keys=all_features_f32,
                        values=all_features_f32,
                    )
                    if self.training:
                        _dbg.info(
                            f"[sample {b}] cross_attn[{_layer_idx}]: "
                            f"queries abs_max={queries.abs().max().item():.4f}, "
                            f"has_nan={torch.isnan(queries).any().item()}, "
                            f"has_inf={torch.isinf(queries).any().item()}"
                        )
                queries = self.query_norm(queries)

            # --- Per-slot predictions ---
            per_slot_scores, per_slot_distances, per_slot_cls, per_slot_logits = (
                self._compute_per_slot_predictions(all_features, queries)
            )

            # --- Loss ---
            if self.training or origin_coord_all is not None:
                sample_input = {"origin_coord": origin_coord_all[b]}
                if origin_type_all is not None:
                    if origin_type_all.dim() == 0:
                        sample_input["origin_type"] = origin_type_all
                    else:
                        sample_input["origin_type"] = origin_type_all[b]
                loss = self._single_fragment_loss(
                    per_slot_scores[0],
                    per_slot_logits[0],
                    per_slot_distances[0] if per_slot_distances else None,
                    per_slot_cls[0],
                    all_coords,
                    sample_input,
                )
                batch_losses.append(loss)

            # --- Eval outputs ---
            if not self.training:
                N_real = sample_features.shape[0]
                batch_eval_outputs.append({
                    "origin_scores": per_slot_scores[0][:N_real],
                    "all_coords": sample_coords,
                    "origin_class": per_slot_cls[0].argmax(dim=-1),
                    "origin_distances": (
                        per_slot_distances[0][:N_real]
                        if per_slot_distances else None
                    ),
                })

        # ==================================================================
        # Aggregate across batch
        # ==================================================================
        if batch_losses:
            return_dict["loss"] = torch.stack(batch_losses).mean()

        if not self.training and batch_eval_outputs:
            if len(batch_eval_outputs) == 1:
                # Single sample: flat tensors (backward compatible)
                return_dict["origin_scores"] = batch_eval_outputs[0]["origin_scores"]
                return_dict["all_coords"] = batch_eval_outputs[0]["all_coords"]
                return_dict["origin_class"] = batch_eval_outputs[0]["origin_class"]
                if batch_eval_outputs[0]["origin_distances"] is not None:
                    return_dict["origin_distances"] = batch_eval_outputs[0]["origin_distances"]
            else:
                # Multi-sample: per-sample lists + stacked class tensor
                return_dict["origin_scores"] = [
                    o["origin_scores"] for o in batch_eval_outputs
                ]
                return_dict["all_coords"] = [
                    o["all_coords"] for o in batch_eval_outputs
                ]
                return_dict["origin_class"] = torch.stack([
                    o["origin_class"] for o in batch_eval_outputs
                ])
                dists = [o["origin_distances"] for o in batch_eval_outputs]
                if dists[0] is not None:
                    return_dict["origin_distances"] = dists

        return return_dict

    @staticmethod
    def _downsample_mask(mask_input_res, input_coords, output_coords):
        """Map a per-input-point mask to per-output-point via nearest-neighbor.

        Uses chunked distance computation to avoid allocating a full
        (N_out, N_in) matrix which can be ~400MB at N=10K.
        """
        if mask_input_res.shape[0] == output_coords.shape[0]:
            return mask_input_res  # already at output resolution
        out_f = output_coords.float()
        in_f = input_coords.float()
        chunk_size = 1024
        nn_idx = torch.empty(out_f.shape[0], dtype=torch.long,
                             device=out_f.device)
        for start in range(0, out_f.shape[0], chunk_size):
            end = min(start + chunk_size, out_f.shape[0])
            dists = torch.cdist(
                out_f[start:end].unsqueeze(0),
                in_f.unsqueeze(0),
            ).squeeze(0)
            nn_idx[start:end] = dists.argmin(dim=1)
        return mask_input_res[nn_idx]

    def _compute_shower_centroids(self, real_coords, shower_mask):
        """
        Compute centroid for the single shower fragment to place dense grid.

        Args:
            real_coords: (N, 3) point coordinates
            shower_mask: (N,) boolean mask, or None

        Returns:
            centroids: list of (3,) tensors
        """
        centroids = []
        if shower_mask is not None:
            m_bool = shower_mask.bool()
            if m_bool.any():
                centroids.append(real_coords[m_bool].mean(dim=0))
        return centroids

    def _compute_per_slot_predictions(self, event_features, queries):
        """
        Compute origin predictions for each slot independently.

        For each slot k:
        - Broadcast the slot query to all points
        - Concatenate with point features
        - Pass through combiner and score head
        - Predict inside/outside classification from the slot itself

        Args:
            event_features: (N_total, D) features for all points (real + virtual)
            queries: (K, D) refined slot representations

        Returns:
            per_slot_scores: list of K (N_total,) score tensors (sigmoid-applied)
            per_slot_distances: list of K (N_total,) distance tensors (or empty list)
            per_slot_cls: list of K (num_classes,) classification logit tensors
            per_slot_logits: list of K (N_total,) raw logits (for numerically stable BCE)
        """
        K = queries.shape[0]
        N = event_features.shape[0]

        per_slot_scores = []
        per_slot_distances = []
        per_slot_cls = []
        per_slot_logits = []

        import logging as _logging
        _dbg = _logging.getLogger("nan_debug")

        for k in range(K):
            # Broadcast slot query to all points
            slot_query = queries[k:k + 1]  # (1, D)
            broadcast = slot_query.expand(N, -1)  # (N, D)

            if self.training:
                _dbg.info(
                    f"[slot {k}] query abs_max={slot_query.abs().max().item():.4f}, "
                    f"event_feats abs_max={event_features.abs().max().item():.4f}, "
                    f"N={N}"
                )

            # Combine with point features — run in float32 to avoid
            # bfloat16 overflow with large input dim (2*D=2176)
            with torch.amp.autocast('cuda', enabled=False):
                combined = torch.cat([event_features.float(), broadcast.float()], dim=-1)  # (N, 2D)
                combined = self.combiner(combined)  # (N, D)

            # Score and distance prediction
            pred = self.score_head(combined)
            per_slot_scores.append(pred["scores"])  # (N,)
            per_slot_logits.append(pred["score_logits"])  # (N,) raw logits
            if "distances" in pred:
                per_slot_distances.append(pred["distances"])  # (N,)

            # Classification: origin type (INSIDE/OUTSIDE/ON_TRACK)
            cls_logits = self.classification_head(queries[k])  # (num_classes,)
            per_slot_cls.append(cls_logits)

        return per_slot_scores, per_slot_distances, per_slot_cls, per_slot_logits

    def _single_fragment_loss(
        self, scores, logits, distances, cls_logits, coords, input_dict,
    ):
        """
        Compute direct loss for a single fragment (K=1, no Hungarian matching).

        Args:
            scores: (N_total,) sigmoid scores from slot 0
            logits: (N_total,) raw logits from slot 0
            distances: (N_total,) predicted distances, or None
            cls_logits: (num_classes,) classification logits from slot 0
            coords: (N_total, 3) point coordinates (real + virtual)
            input_dict: dict with origin_coord, origin_type, origin_distance

        Returns:
            total_loss: scalar loss
        """
        device = coords.device
        origin_coord = input_dict["origin_coord"]  # (3,)

        # Gaussian soft labels from distance to origin
        distances_to_origin = torch.norm(
            coords - origin_coord.unsqueeze(0), dim=-1
        )
        target_scores = torch.exp(
            -0.5 * (distances_to_origin / self.gaussian_sigma) ** 2
        )

        # Score loss: BCE with logits for numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            safe_logits = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=50.0, neginf=-50.0
            )
            score_loss = F.binary_cross_entropy_with_logits(
                safe_logits,
                target_scores.float(),
                reduction='mean',
            )

        total_loss = self.score_loss_weight * score_loss

        # Classification loss (cross-entropy)
        origin_type = input_dict.get("origin_type", None)
        if origin_type is not None:
            ot = origin_type.long()
            if ot.dim() == 0:
                ot = ot.unsqueeze(0)
            with torch.amp.autocast('cuda', enabled=False):
                safe_cls = torch.nan_to_num(
                    cls_logits.float(), nan=0.0
                ).unsqueeze(0)
                cls_loss = F.cross_entropy(safe_cls, ot.to(device))
            total_loss = total_loss + self.classification_loss_weight * cls_loss

        # Distance loss (smooth L1)
        if distances is not None:
            with torch.amp.autocast('cuda', enabled=False):
                safe_dist = torch.nan_to_num(
                    distances.float(), nan=0.0
                )
                dist_loss = F.smooth_l1_loss(
                    safe_dist,
                    distances_to_origin.float(),
                    reduction='mean',
                )
            total_loss = total_loss + self.distance_loss_weight * dist_loss

        return total_loss
