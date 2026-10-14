"""
Electronics clocks, detector properties and LAr properties.

Names follow larlite's `TimeService`, `DetectorProperties` and `LArProperties`
(and the LArSoft providers they wrap). Values are the job defaults of the
extracted services configuration (see /provenance in the HDF5 file).
"""
from dataclasses import dataclass

import numpy as np

from .io import read_attrs, read_table


@dataclass(frozen=True)
class ElecClock:
    """Like detinfo::ElecClock: period in us, frequency in MHz."""
    TickPeriod: float
    Frequency: float
    FramePeriod: float
    FrameTicks: float

    def Ticks(self, time_us):
        return np.asarray(time_us) / self.TickPeriod


class DetectorClocks:
    def __init__(self, h5file):
        self.attrs = read_attrs(h5file["clocks"])
        a = self.attrs

        def clk(name):
            return ElecClock(a[f"{name}_TickPeriod_us"], a[f"{name}_Frequency_MHz"],
                             a[f"{name}_FramePeriod_us"], a[f"{name}_FrameTicks_ticks"])
        self._tpc = clk("TPCClock")
        self._opt = clk("OpticalClock")
        self._trig = clk("TriggerClock")
        self._ext = clk("ExternalClock")

    def TPCClock(self):
        return self._tpc

    def OpticalClock(self):
        return self._opt

    def TriggerClock(self):
        return self._trig

    def ExternalClock(self):
        return self._ext

    def TriggerTime(self):
        return self.attrs["TriggerTime_us"]

    def BeamGateTime(self):
        return self.attrs["BeamGateTime_us"]

    def TriggerOffsetTPC(self):
        """TPC readout start relative to the trigger [us] (negative = before)."""
        return self.attrs["TriggerOffsetTPC_us"]

    def G4ToElecTime(self, g4_time_ns=0.0):
        """G4 time [ns] -> electronics time [us] (job-default trigger)."""
        return self.attrs["G4ToElecTime_at_g4t0_us"] + np.asarray(g4_time_ns) * 1e-3

    def TPCTick2TrigTime(self, tick):
        """TPC tick -> time relative to trigger [us]."""
        return self.attrs["TPCTick2TrigTime_at_tick0_us"] + np.asarray(tick) * self._tpc.TickPeriod

    def TPCTick2Time(self, tick):
        """TPC tick -> electronics time [us]."""
        return self.attrs["TPCTick2Time_at_tick0_us"] + np.asarray(tick) * self._tpc.TickPeriod


class DetectorProperties:
    def __init__(self, h5file):
        g = h5file["detector_properties"]
        self.attrs = read_attrs(g)
        xt = read_table(g["x_to_ticks"])
        self._xt = {(int(c), int(t), int(p)): (float(a), float(b))
                    for c, t, p, a, b in zip(xt["cryo"], xt["tpc"], xt["plane"],
                                             xt["ticks_per_cm"], xt["tick_at_x0"])}
        self._tick_period = read_attrs(h5file["clocks"])["TPCClock_TickPeriod_us"]

    def SamplingRate(self):
        """TPC tick period in ns (larlite convention)."""
        return 1000.0 * self._tick_period

    def NumberTimeSamples(self):
        return int(self.attrs["NumberTimeSamples"])

    def ReadOutWindowSize(self):
        return int(self.attrs["ReadOutWindowSize"])

    def ElectronsToADC(self):
        return self.attrs["ElectronsToADC"]

    def Temperature(self):
        return self.attrs["Temperature_K"]

    def ElectronLifetime(self):
        return self.attrs["ElectronLifetime_us"]

    def Density(self):
        return self.attrs["Density_g_cm3"]

    def Efield(self, planegap=0):
        return float(np.asarray(self.attrs["Efield_kV_cm"])[planegap])

    def DriftVelocity(self):
        """Nominal drift velocity [cm/us] at Efield(0), Temperature()."""
        return self.attrs["DriftVelocity_cm_us"]

    def ConvertXToTicks(self, x, plane, tpc=0, cryo=0):
        a, b = self._xt[(cryo, tpc, plane)]
        return a * np.asarray(x, dtype=np.float64) + b

    def ConvertTicksToX(self, ticks, plane, tpc=0, cryo=0):
        a, b = self._xt[(cryo, tpc, plane)]
        return (np.asarray(ticks, dtype=np.float64) - b) / a

    def GetXTicksCoefficient(self, plane=0, tpc=0, cryo=0):
        """ticks per cm (signed by drift direction)."""
        return self._xt[(cryo, tpc, plane)][0]

    def GetXTicksOffset(self, plane, tpc=0, cryo=0):
        """tick at x=0."""
        return self._xt[(cryo, tpc, plane)][1]


class LArProperties:
    """Scalars are available as methods named like LArSoft's
    (e.g. RadiationLength(), ScintYield()), spectra as dicts of arrays."""

    def __init__(self, h5file):
        g = h5file["lar_properties"]
        self.attrs = read_attrs(g)
        self.spectra = {k: (g["spectra"][k]["energy_eV"][()], g["spectra"][k]["value"][()])
                        for k in g["spectra"]} if "spectra" in g else {}
        dp = read_attrs(h5file["detector_properties"])
        self._temperature = dp["Temperature_K"]
        self._density = dp["Density_g_cm3"]
        self._efield = np.asarray(dp["Efield_kV_cm"])

    def __getattr__(self, name):
        attrs = self.__dict__.get("attrs", {})
        if name in attrs:
            return lambda *a, **k: attrs[name]
        spectra = self.__dict__.get("spectra", {})
        if name in spectra:
            return lambda *a, **k: spectra[name]
        raise AttributeError(name)

    def Temperature(self):
        return self._temperature

    def Density(self):
        return self._density

    def Efield(self, planegap=0):
        return float(self._efield[planegap])
