"""ID types and enums mirroring LArSoft's geo:: namespace (zero-based indices)."""
from enum import IntEnum
from typing import NamedTuple


class CryostatID(NamedTuple):
    cryo: int


class TPCID(NamedTuple):
    cryo: int
    tpc: int


class PlaneID(NamedTuple):
    cryo: int
    tpc: int
    plane: int


class WireID(NamedTuple):
    cryo: int
    tpc: int
    plane: int
    wire: int


class View(IntEnum):
    """geo::View_t"""
    kU = 0
    kV = 1
    kZ = 2          # == kW (collection plane in SBND/MicroBooNE/ICARUS)
    kY = 3
    kX = 4
    k3D = 5
    kUnknown = 6


class SigType(IntEnum):
    """geo::SigType_t"""
    kInduction = 0
    kCollection = 1
    kMysteryType = 2
