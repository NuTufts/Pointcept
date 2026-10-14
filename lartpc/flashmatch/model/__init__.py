"""
Per-detector flash model (see ../SBND_SEMIANALYTIC_PLAN.md).

    from lartpc.flashmatch.model import make_flash_model
    fm = make_flash_model("uboone")                       # today's MicroBooNE behaviour
    pe = fm.predict(pos_cm, charge, cluster_id, n_clusters, flash_t0_us, light_yield=G)
    chi2 = fm.chi2(pe[0], obs_pe, mask=fm.channel_mask(obs_pe, run))
"""
from .base import FlashModel, ScoreConfig, normalized_chi2  # noqa: F401


def make_flash_model(detector, **kw):
    detector = detector.lower()
    if detector in ("uboone", "microboone"):
        from .detectors.uboone import UBooNEFlashModel
        return UBooNEFlashModel(**kw)
    raise ValueError(f"no flash model for detector {detector!r} (available: uboone)")
