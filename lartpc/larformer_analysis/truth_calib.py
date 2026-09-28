"""Truth-side visible (deposited) energy calibration for the gen2ntuple
`trueSimPartPixelSumQ` branch -- ONE place for the constants every analysis
should import (pi0, nue, sigma0, single_photon ...).

`trueSimPartPixelSumQ` is the de-double-counted comb charge (Y else mean(U,V))
of the merged_sp spacepoints truth-matched to a particle, UNCALIBRATED. The
deposited energy is  E_dep = a_species x Q.

MEASURED 2026-09-27 (single-photon review) on well-contained true particles
(conversion/start point >= 40 cm from every TPC wall, 30-500 MeV), median
charge per MeV, identical on the run-1 and run-3 label-completed overlays:
    photon   90.8 / 91.6 ADC/MeV  ->  A_GAMMA_DEP = 0.0110 MeV/ADC
    electron 98.9 / 99.3 ADC/MeV  ->  A_E_DEP     = 0.0101 MeV/ADC
Cross-check: reconstructed showerRecoE (recal3) over A_GAMMA_DEP x Q has a
median of 1.04, flat within +-10% from 40 to 500 MeV, so truth and reco
energies are now on the same scale; recoE / true initial E = 0.88, the
remainder being containment.

LEGACY: 0.0253017 MeV/ADC (`calo_calib_pre_photon_recal.npz` gamma_a) was a
July-2026 RECO cluster fit on the old chain before label expansion and before
the affine recalibration; applied to the (expanded) truth charge it overstates
deposited energy by x2.30, so "E_vis >= 20 MeV" meant "deposited >= 8.7 MeV"
while the reco cut demanded 20 MeV. Kept only to reproduce old results.
"""
A_GAMMA_DEP = 0.0110      # MeV/ADC, photons (deposited energy)
A_E_DEP = 0.0101          # MeV/ADC, electrons (deposited energy)
A_GAMMA_LEGACY = 0.0253017
EVIS_MIN = 20.0           # MeV deposited: the detectability threshold

SCALES = {"deposited": (A_GAMMA_DEP, A_E_DEP),
          "legacy": (A_GAMMA_LEGACY, A_GAMMA_LEGACY)}


def em_constants(scale="deposited"):
    """(a_gamma, a_e) for the named truth energy scale."""
    return SCALES[scale]
