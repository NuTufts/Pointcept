"""
Shower Origin Prediction Models

Query-conditioned architecture for identifying electromagnetic shower
origin points in LArTPC data using Slot Attention and cross-attention.
"""

from .shower_origin_model import (
    ShowerOriginPredictor,
    ShowerOriginPredictorV2,
    ShowerOriginPredictorV3,
    OriginClassificationHead,
)
from .slot_attention import SlotAttention, MultiHeadSlotAttention
from .virtual_grid import VirtualGridGenerator, PositionalEncoding3D

__all__ = [
    "ShowerOriginPredictor",
    "ShowerOriginPredictorV2",
    "ShowerOriginPredictorV3",
    "OriginClassificationHead",
    "SlotAttention",
    "MultiHeadSlotAttention",
    "VirtualGridGenerator",
    "PositionalEncoding3D",
]
