"""Per-event LArFormer analysis — computes M1 (IoU), M3 (chi-2), M4 (rank).

Inputs (per event):
  --merged-h5      MicroBooNE merged H5 (GT triplet + slice info)
  --flashinfo-h5   Extended flashinfo H5 (in-time flash + event_truth)
  --inference-h5   slicerpred_*.h5 from tools/run_slicer_inference.py
  --output-h5      Where to write the per-event metrics H5

Output: a single `perevent_<run>_<sub>_<evt>.h5` (or, when `--fileno` and
`--entry` are also given, `perevent_fileno<NNNNN>_entry<NNNNNN>_run<R>_sub<S>_evt<E>.h5`)
with the schema below.
The aggregator (`aggregate_metrics.py`) reads N of these and emits the
flat `event_summary.h5`.

What it computes:

  GT side
  -------
    - GT-nu mask = union of SPs (in post/coord space) whose GT slice
      has origin_type == nu (after the slice_class_map={1:0, 2:1} remap
      → origin_type == 0). For merge_nu_slices=True configs this is the
      whole nu interaction in one go.
    - In-time flash = first beam-producer (producer_id==0) flash that
      flashinfo paired with a GT-nu primary's slice. We carry both its
      pe (observed) and t0_us.
    - GT-baseline PE = PhotonLib prediction from the GT-nu union SPs
      (drift-corrected to the in-time flash's t0).

  Predicted side
  --------------
    - Per predicted slice (= one query that won ≥1 SP after the
      panoptic argmax in `post/pred_slice_id`):
        * IoU vs the GT-nu mask
        * class_argmax (nu / cosmic / no_object)
        * OOB fraction after drift correction
        * Neyman chi-2 vs the in-time flash at each OOB threshold
          (NaN where oob_frac > threshold)
        * Predicted PE vector (32,)
    - M1 = max IoU among predicted slices labeled nu by the model
           (queries with class_argmax==nu). NaN/0 if no nu pred.
    - M3 = chi-2 of the M1 slice at the default OOB threshold;
           delta_chi2[t] = chi2_M1[t] - chi2_GT[t] for each threshold.
    - M4_all = rank (1-based) of the *GT-best-match* predicted slice
               (the slice with max IoU vs GT-nu — not class-filtered)
               when ALL non-rejected slices are sorted by chi-2 asc.
    - M4_nu  = same restricted to model-nu slices.

Defaults are tuned to match `tools/visualize_slice_flash_match.py`'s
γ + readout-factor conventions. Pass --gamma-beam / --gamma-cosmic to
override.

Usage:
  python analyze_event.py \\
      --merged-h5    /path/to/merged_*.h5 \\
      --flashinfo-h5 /path/to/flashinfo_*.h5 \\
      --inference-h5 /path/to/slicerpred_*.h5 \\
      --output-h5    /path/to/perevent_<run>_<sub>_<evt>.h5 \\
      [--model-tag crosslevelrefiner_mixedq_maskdn] \\
      [--gamma-beam 1.0 --gamma-cosmic 1.0]
"""

import argparse
import os
import sys

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from lartpc.larformer_analysis.slicer_eval.lib import (  # noqa: E402
    categorize, flash_chi2, flash_predict,
)


DEFAULT_OOB_THRESHOLDS = np.array([0.0, 0.05, 0.10, 0.20, 0.50],
                                  dtype=np.float32)
DEFAULT_OOB_IDX = 3  # index in DEFAULT_OOB_THRESHOLDS for headline (0.20)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_inference(p):
    """Slice the inference H5 into a single per-event dict."""
    with h5py.File(p, "r") as f:
        out = dict(
            pre_coord        = f["pre/coord"][:].astype(np.float32),
            post_coord       = f["post/coord"][:].astype(np.float32),
            post_pred_query  = f["post/pred_query"][:].astype(np.int64),
            post_pred_class  = f["post/pred_class"][:].astype(np.int64),
            post_pred_slice  = f["post/pred_slice_id"][:].astype(np.int64),
            post_slice_gt    = f["post/slice_id_gt"][:].astype(np.int64),
            pre_pixval       = f["pre/pixval"][:].astype(np.float32),
            pre_keep         = f["pre/keep"][:].astype(bool),
            q_class_argmax   = f["queries/class_argmax"][:].astype(np.int64),
            gt_origin_type   = f["gt/origin_type"][:].astype(np.int64),
            gt_primary_trackid = f["gt/primary_trackid"][:].astype(np.int64),
            # Pre-/post-argmax pair info (drives the overclaim metrics).
            # See docs at the top of this file. Required: present since
            # the v2 inference output (run_slicer_inference.py 2026-05-19+).
            gt_matched_query = f["gt/matched_query"][:].astype(np.int64),
            gt_pair_iou      = f["gt/pair_iou"][:].astype(np.float32),
            gt_argmax_iou    = f["gt/argmax_iou"][:].astype(np.float32),
        )
        # Per-event metadata.
        out["run"] = int(f.attrs.get("meta_run", -1))
        out["subrun"] = int(f.attrs.get("meta_subrun", -1))
        out["event"] = int(f.attrs.get("meta_event", -1))
        out["no_object_class_id"] = int(f.attrs.get("meta_no_object_class_id", 2))
    out["post_pixval"] = out["pre_pixval"][out["pre_keep"]]
    return out


def _load_flashinfo(p):
    """Slice the flashinfo H5 into a single per-event dict."""
    with h5py.File(p, "r") as f:
        e0 = f["entry_0"]
        out = dict(
            run    = int(e0.attrs.get("run", -1)),
            subrun = int(e0.attrs.get("subrun", -1)),
            event  = int(e0.attrs.get("event", -1)),
            n_pmts = int(e0.attrs.get("n_pmts", 32)),
            flash_pe          = e0["flashes/pe"][:].astype(np.float32),
            flash_time_us     = e0["flashes/time_us"][:].astype(np.float32),
            flash_producer_id = e0["flashes/producer_id"][:].astype(np.int32),
            flash_matched_slice = e0["flashes/matched_slice_id"][:].astype(np.int64),
            slice_id              = e0["slice_flash_matches/slice_id"][:].astype(np.int64),
            slice_primary_origin  = e0["slice_flash_matches/primary_origin"][:].astype(np.int32),
            slice_matched_flash_idx = e0["slice_flash_matches/matched_flash_idx"][:].astype(np.int32),
        )
        # Categorize event right here so we don't pass the file handle around.
        out["category_mask"] = int(categorize.categorize_event(e0))
        out["n_visible_nu_gammas"] = int(categorize.count_visible_nu_gammas(e0))
        out["n_primary_pi0"] = int(categorize.count_primary_pi0(e0))
        if "event_truth" in e0:
            et = e0["event_truth"]
            out["truth"] = dict(
                has_neutrino = bool(et.attrs.get("has_neutrino", False)),
                nu_pdg       = int(et.attrs.get("nu_pdg", 0)),
                nu_energy_MeV = float(et.attrs.get("nu_energy_MeV", 0.0)),
                ccnc         = int(et.attrs.get("ccnc", -1)),
                mode         = int(et.attrs.get("mode", -1)),
                interaction_type = int(et.attrs.get("interaction_type", -1)),
            )
        else:
            out["truth"] = dict(
                has_neutrino=False, nu_pdg=0, nu_energy_MeV=0.0,
                ccnc=-1, mode=-1, interaction_type=-1,
            )
    return out


# ---------------------------------------------------------------------------
# Merged-H5 per-point join + pixel-share normalization
# ---------------------------------------------------------------------------
#
# The charge fed to the flash predictor must be the RAW Y-plane ADC (linear
# in deposited charge) — NOT the inference H5's `pre/pixval`, which is the
# log10-transformed, [-1,1]-normalized backbone feature. The raw ADC, plus
# the per-SP wire/tick image coordinates needed to detect when several SPs
# project into the same wire pixel, live only in the merged H5. We recover
# them by joining the inference `pre/coord` (exact detector cm — a pure
# row-subset of the merged triplet `pos`) back to the merged rows, then
# index by `pre/keep` to align with the post-SPs.

def _load_merged_perpoint(merged_path, inf):
    """Recover raw-ADC pixval + (tick, u/v/y wire) per post-SP.

    Joins inference `pre/coord` → merged-H5 triplet rows by exact float
    coordinate, then applies `pre/keep` to align with the post-SP order.

    Returns a dict of post-SP-aligned arrays, or None if any pre-SP fails
    to match (caller then falls back to the log-normalized `pre/pixval`
    charge and disables share-normalization).
    """
    with h5py.File(merged_path, "r") as f:
        td = f["entry_0"]["triplet_data"]
        pos    = td["pos"][:].astype(np.float32)
        rawpix = td["pixval"][:].astype(np.float32)
        uwire  = td["uwire"][:].astype(np.int32)
        vwire  = td["vwire"][:].astype(np.int32)
        ywire  = td["ywire"][:].astype(np.int32)
        tick   = td["tick"][:].astype(np.int32)

    # Exact-float coordinate join. `post/coord` is affine-reconstructed and
    # does NOT match exactly — must go through `pre/coord` + `pre/keep`.
    row_of = {}
    for i in range(pos.shape[0]):
        kb = pos[i].tobytes()
        if kb not in row_of:
            row_of[kb] = i
    pre_coord = inf["pre_coord"]
    pre_idx = np.fromiter(
        (row_of.get(pre_coord[i].tobytes(), -1)
         for i in range(pre_coord.shape[0])),
        dtype=np.int64, count=pre_coord.shape[0],
    )
    if (pre_idx < 0).any():
        return None
    post_idx = pre_idx[inf["pre_keep"]]
    return dict(
        post_rawpix = rawpix[post_idx],
        post_uwire  = uwire[post_idx],
        post_vwire  = vwire[post_idx],
        post_ywire  = ywire[post_idx],
        post_tick   = tick[post_idx],
    )


def _pixel_share_counts(rawpix, tick, uwire, ywire, cluster_id):
    """Per-SP count of SPs sharing the same source wire-pixel within the
    same cluster.

    The 'source pixel' is the plane that supplied the SP's charge in
    `select_charge_y_with_uv_fallback_np`: (tick, ywire) for Y-selected
    SPs, (tick, uwire) for the (rare, dead-Y) UV-fallback SPs. The
    `used_uv` flag is part of the key so Y and UV groups never merge.

    Dividing the selected charge by this count keeps each wire pixel's
    ADC integral constant within the cluster — so absorbing ghost SPs
    that land on already-lit pixels does not inflate the slice's total
    charge.
    """
    used_uv = rawpix[:, 2] <= 0
    wire = np.where(used_uv, uwire, ywire).astype(np.int64)
    keys = np.stack([
        cluster_id.astype(np.int64),
        used_uv.astype(np.int64),
        tick.astype(np.int64),
        wire,
    ], axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                               return_counts=True)
    return counts[inv]


# ---------------------------------------------------------------------------
# In-time flash + GT-nu mask
# ---------------------------------------------------------------------------

def _find_in_time_flash(fi):
    """Return (flash_idx, pe_obs, t0_us, producer_id, paired_slice_id) for
    the in-time flash, or None if none found.

    Definition: the (beam-producer) flash that flashinfo paired with the
    first GT-nu primary slice (primary_origin == 1) that has a matched
    flash. This is what `visualize_slice_flash_match.py` calls the
    'in-time matched flash' for the nu interaction.

    Fallback: if no GT-nu slice is matched to any flash, return the
    nearest-to-t=0 beam-producer flash (rare; happens on out-of-time-only
    events).
    """
    so = fi["slice_primary_origin"]
    sf = fi["slice_matched_flash_idx"]
    sid = fi["slice_id"]
    nu_match = (so == 1) & (sf >= 0)
    if nu_match.any():
        which = int(np.flatnonzero(nu_match)[0])
        fidx = int(sf[which])
        return dict(
            flash_idx        = fidx,
            pe_obs           = fi["flash_pe"][fidx].astype(np.float32),
            t0_us            = float(fi["flash_time_us"][fidx]),
            producer_id      = int(fi["flash_producer_id"][fidx]),
            paired_slice_id  = int(sid[which]),
        )
    # Fallback: nearest-to-t=0 beam flash.
    beam = (fi["flash_producer_id"] == 0)
    if beam.any():
        idxs = np.flatnonzero(beam)
        fidx = int(idxs[np.argmin(np.abs(fi["flash_time_us"][idxs]))])
        return dict(
            flash_idx        = fidx,
            pe_obs           = fi["flash_pe"][fidx].astype(np.float32),
            t0_us            = float(fi["flash_time_us"][fidx]),
            producer_id      = int(fi["flash_producer_id"][fidx]),
            paired_slice_id  = -1,
        )
    return None


def _gt_nu_sp_mask(inf):
    """Bool mask over post-SPs that are part of any nu-origin GT instance.

    Uses the inference H5's:
      gt/origin_type[k] == 0 → nu (after slice_class_map={1:0, 2:1} remap)
      gt/primary_trackid[k] → the slice_id that post/slice_id_gt refers to
    """
    nu_gt_idx = np.flatnonzero(inf["gt_origin_type"] == 0)
    if len(nu_gt_idx) == 0:
        return np.zeros(inf["post_coord"].shape[0], dtype=bool)
    nu_trackids = set(int(t) for t in inf["gt_primary_trackid"][nu_gt_idx])
    sp_gt = inf["post_slice_gt"]
    mask = np.zeros(sp_gt.shape[0], dtype=bool)
    for tid in nu_trackids:
        mask |= (sp_gt == tid)
    return mask


# ---------------------------------------------------------------------------
# Per-predicted-slice computation
# ---------------------------------------------------------------------------

def _compute_pred_slices(
    inf, in_time, gt_nu_mask,
    gamma_by_producer, readout_factor_by_producer,
    photonlib_cache, oob_thresholds, f_sys, eps,
    v_drift, charge_source="raw_adc", share_norm="none",
):
    """Returns a dict of (Q,) arrays describing every non-empty predicted
    slice in the event, sorted by query_id ascending.
    """
    no_obj = inf["no_object_class_id"]
    # Group SPs by the panoptic-argmax-winning QUERY ID. Earlier
    # versions grouped by `post/pred_slice_id` — that field is the
    # GT primary_trackid for matched queries (a large arbitrary
    # number), NOT the query id. Using it broke the class_argmax
    # lookup (out-of-bounds → silent fallback to no_object) and
    # made every panoptic slice appear as 'no_object', spuriously
    # producing n_pred_nu_slices=0 for events where the model
    # actually predicted nu correctly. See README §"Over-claim".
    pred_query = inf["post_pred_query"]
    # Filter out SPs whose winning query class is no_object — those
    # are "unassigned" panoptic SPs and shouldn't appear as slices.
    # The pred_class array is already cls_argmax[pred_query], so this
    # is equivalent to "active queries only".
    pred_class_sp = inf["post_pred_class"]
    valid_sp_mask = (pred_class_sp != no_obj)
    if not valid_sp_mask.any():
        return dict(
            query_id     = np.zeros(0, dtype=np.int32),
            class_argmax = np.zeros(0, dtype=np.int32),
            n_sp         = np.zeros(0, dtype=np.int32),
            iou_vs_gt_nu = np.zeros(0, dtype=np.float32),
            oob_frac     = np.zeros(0, dtype=np.float32),
            chi2         = np.zeros((0, len(oob_thresholds)), dtype=np.float32),
            pe_pred      = np.zeros((0, 32), dtype=np.float32),
        )
    uniq_q = np.sort(np.unique(pred_query[valid_sp_mask]))
    n_q = len(uniq_q)

    have_raw = (charge_source == "raw_adc" and "post_rawpix" in inf)
    pos_all = inf["post_coord"]
    px_all  = inf["post_rawpix"] if have_raw else inf["post_pixval"]

    # Build per-cluster cluster_id for batched PhotonLib call:
    # global SP-level cluster_id mapping uniq_q → [0, n_q).
    q_to_cid = {int(q): ci for ci, q in enumerate(uniq_q)}
    # Filter to valid_sp_mask SPs only.
    sp_pos    = pos_all[valid_sp_mask]
    sp_px     = px_all[valid_sp_mask]
    sp_pred_q = pred_query[valid_sp_mask]
    sp_gt_nu  = gt_nu_mask[valid_sp_mask]
    sp_cid    = np.array([q_to_cid[int(q)] for q in sp_pred_q], dtype=np.int64)
    sp_charge = flash_predict.select_charge_y_with_uv_fallback_np(sp_px)
    # Per-slice pixel-share normalization: divide each SP's charge by the
    # number of SPs in the SAME slice sharing its source wire pixel, so the
    # slice's total charge equals its 2D pixel-footprint integral.
    if share_norm == "per-slice" and have_raw:
        share = _pixel_share_counts(
            sp_px, inf["post_tick"][valid_sp_mask],
            inf["post_uwire"][valid_sp_mask], inf["post_ywire"][valid_sp_mask],
            sp_cid,
        )
        sp_charge = sp_charge / share.astype(np.float32)

    # Batched PE prediction.
    pe_pred_all = flash_predict.predict_many_slices_pe(
        pos_cm=sp_pos, charge=sp_charge, cluster_id=sp_cid,
        n_clusters=n_q, flash_t0_us=in_time["t0_us"],
        producer_id=in_time["producer_id"],
        gamma_by_producer=gamma_by_producer,
        readout_factor_by_producer=readout_factor_by_producer,
        photonlib_cache=photonlib_cache,
        v_drift_cm_per_us=v_drift,
    )

    # Per-slice scalars.
    class_argmax = np.full(n_q, no_obj, dtype=np.int32)
    n_sp_arr     = np.zeros(n_q, dtype=np.int32)
    iou_arr      = np.zeros(n_q, dtype=np.float32)
    oob_frac_arr = np.zeros(n_q, dtype=np.float32)
    chi2_arr     = np.full((n_q, len(oob_thresholds)), np.nan, dtype=np.float32)

    gt_nu_total = int(gt_nu_mask.sum())
    for ci, q in enumerate(uniq_q):
        q = int(q)
        class_argmax[ci] = int(inf["q_class_argmax"][q]) if 0 <= q < len(inf["q_class_argmax"]) else no_obj
        member = (sp_cid == ci)
        n_sp_arr[ci] = int(member.sum())
        # IoU vs GT-nu mask (over the full post-SP space — note GT-nu may
        # include SPs NOT assigned to this query but still part of the
        # GT nu interaction).
        pred_total = n_sp_arr[ci]
        pred_in_gt = int(sp_gt_nu[member].sum())
        union = gt_nu_total + pred_total - pred_in_gt
        iou_arr[ci] = float(pred_in_gt / union) if union > 0 else 0.0
        # OOB + chi-2 sweep (computes drift correction internally).
        sub = flash_chi2.chi2_with_oob(
            pos_cm=sp_pos[member],
            pe_pred=pe_pred_all[ci],
            pe_obs=in_time["pe_obs"],
            flash_t0_us=in_time["t0_us"],
            oob_thresholds=oob_thresholds,
            f_sys=f_sys, eps=eps,
        )
        oob_frac_arr[ci] = sub["oob_frac"]
        chi2_arr[ci, :]  = sub["chi2"]

    return dict(
        query_id     = uniq_q.astype(np.int32),
        class_argmax = class_argmax,
        n_sp         = n_sp_arr,
        iou_vs_gt_nu = iou_arr,
        oob_frac     = oob_frac_arr,
        chi2         = chi2_arr,
        pe_pred      = pe_pred_all.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics (M1, M3, M4)
# ---------------------------------------------------------------------------

def _compute_overclaim(inf, nu_class_id=0):
    """Quantify the panoptic-argmax vs per-pair view discrepancy.

    The model's `class_argmax` per query can be correct (== nu) while
    that same query LOSES the panoptic per-SP argmax to competitors —
    in which case the analyzer's downstream view (which uses panoptic
    slices) never sees the correct nu prediction. See README §"Over-
    claim metrics" and the diagnostic walkthrough on entry 1.

    Returns a dict with the per-event headlines + per-nu-GT arrays
    documented in the per-event H5 `overclaim/` schema.

    Definitions:
      sp_level_nu_recall    — frac of GT-nu post-SPs labeled nu by
                              the panoptic argmax (post/pred_class).
      sp_level_nu_precision — frac of panoptic-nu post-SPs that are
                              truly in any nu GT.
      n_nu_gt_instances     — count of GT instances with origin_type==nu.
      n_nu_gt_matched_to_nu_query
                            — of those, count whose Hungarian-matched
                              query has class_argmax == nu.
      n_nu_match_survived_panoptic
                            — of THOSE, count whose matched query won
                              ≥1 SP under panoptic argmax (i.e. appears
                              with a non-(-1) slice id in post/pred_slice_id).
      m1_iou_intent         — pre-argmax matched-query IoU for the best
                              matched-nu query; the "the model thought
                              it found this slice" view.

    Per-nu-GT arrays:
      gt_nu_idx                 — index into gt/origin_type
      matched_query_id          — Hungarian's matched query (or -1)
      matched_query_class       — class_argmax of that query (or -1)
      matched_query_pair_iou    — pre-argmax IoU (= gt/pair_iou)
      matched_query_argmax_iou  — post-argmax IoU (= gt/argmax_iou)
      matched_query_panoptic_n_sp — n SPs won by matched query in panoptic
    """
    gt_origin = inf["gt_origin_type"]
    gt_match  = inf["gt_matched_query"]
    gt_pair   = inf["gt_pair_iou"]
    gt_arg    = inf["gt_argmax_iou"]
    q_cls     = inf["q_class_argmax"]
    post_pred_class = inf["post_pred_class"]
    post_pred_query = inf["post_pred_query"]
    post_slice_gt   = inf["post_slice_gt"]
    nu_trackids = set(int(t) for t in inf["gt_primary_trackid"][gt_origin == nu_class_id])

    # SP-level confusion at the panoptic-argmax view.
    gt_nu_sp_mask  = np.zeros(post_slice_gt.shape[0], dtype=bool)
    for tid in nu_trackids:
        gt_nu_sp_mask |= (post_slice_gt == tid)
    pred_nu_sp_mask = (post_pred_class == nu_class_id)
    n_post_sp_total          = int(post_pred_class.shape[0])
    n_post_sp_gt_nu          = int(gt_nu_sp_mask.sum())
    n_post_sp_pred_nu        = int(pred_nu_sp_mask.sum())
    n_post_sp_true_and_pred  = int((gt_nu_sp_mask & pred_nu_sp_mask).sum())
    sp_recall    = (n_post_sp_true_and_pred / n_post_sp_gt_nu
                    if n_post_sp_gt_nu > 0 else float("nan"))
    sp_precision = (n_post_sp_true_and_pred / n_post_sp_pred_nu
                    if n_post_sp_pred_nu > 0 else float("nan"))

    # Per-nu-GT arrays.
    nu_idxs = np.flatnonzero(gt_origin == nu_class_id)
    K_nu = len(nu_idxs)
    arr_mqid   = np.full(K_nu, -1, dtype=np.int32)
    arr_mcls   = np.full(K_nu, -1, dtype=np.int32)
    arr_piou   = np.full(K_nu, np.nan, dtype=np.float32)
    arr_aiou   = np.full(K_nu, np.nan, dtype=np.float32)
    arr_pansp  = np.zeros(K_nu, dtype=np.int32)
    n_matched_to_nu  = 0
    n_match_survived = 0
    best_intent_iou = 0.0
    best_intent_qid = -1
    for j, k in enumerate(nu_idxs):
        mq = int(gt_match[k])
        arr_mqid[j] = mq
        arr_piou[j] = float(gt_pair[k])
        arr_aiou[j] = float(gt_arg[k])
        if mq < 0:
            continue
        mcls = int(q_cls[mq]) if 0 <= mq < q_cls.shape[0] else -1
        arr_mcls[j] = mcls
        n_won = int((post_pred_query == mq).sum())
        arr_pansp[j] = n_won
        if mcls == nu_class_id:
            n_matched_to_nu += 1
            if n_won > 0:
                n_match_survived += 1
            # m1 intent: pick the highest pair_iou among class-correct
            # matched queries — analogous to the panoptic-view M1 but
            # bypasses the per-SP argmax competition.
            if arr_piou[j] > best_intent_iou:
                best_intent_iou = float(arr_piou[j])
                best_intent_qid = mq

    return dict(
        # Per-event scalars
        sp_level_nu_recall            = float(sp_recall),
        sp_level_nu_precision         = float(sp_precision),
        n_post_sp_total               = int(n_post_sp_total),
        n_post_sp_gt_nu               = int(n_post_sp_gt_nu),
        n_post_sp_pred_nu             = int(n_post_sp_pred_nu),
        n_post_sp_true_and_pred_nu    = int(n_post_sp_true_and_pred),
        n_nu_gt_instances             = int(K_nu),
        n_nu_gt_matched_to_nu_query   = int(n_matched_to_nu),
        n_nu_match_survived_panoptic  = int(n_match_survived),
        m1_iou_intent                 = float(best_intent_iou),
        m1_slice_id_intent            = int(best_intent_qid),
        # Per-nu-GT arrays
        gt_nu_idx                       = nu_idxs.astype(np.int32),
        matched_query_id                = arr_mqid,
        matched_query_class             = arr_mcls,
        matched_query_pair_iou          = arr_piou,
        matched_query_argmax_iou        = arr_aiou,
        matched_query_panoptic_n_sp     = arr_pansp,
    )


def _compute_metrics(pred, gt_baseline, oob_thresholds, default_idx,
                     no_object_class_id, nu_class_id=0):
    """Compute the 4-metric headline + threshold sweeps.

    M1 = max IoU among pred slices with class_argmax == nu_class_id.
    M3 = chi-2 of M1 slice (NaN if no nu pred or rejected at threshold).
    M4_all = rank of GT-best-match slice (max IoU among ALL pred slices,
             ignoring class) when sorted by chi-2 ascending. -1 if the
             best slice is rejected at that threshold.
    M4_nu  = same restricted to nu-class pool. -1 if no nu pred.
    """
    n_th = len(oob_thresholds)
    out = dict(
        has_nu_prediction = False,
        m1_iou       = 0.0,
        m1_slice_id  = -1,
        m3_chi2_gt   = float(gt_baseline["chi2"][default_idx]),
        m3_chi2_nu   = float("nan"),
        m3_delta_chi2 = np.full(n_th, np.nan, dtype=np.float32),
        m4_rank_all  = np.full(n_th, -1, dtype=np.int32),
        m4_rank_nu   = np.full(n_th, -1, dtype=np.int32),
    )
    if pred["query_id"].size == 0:
        return out
    is_nu = (pred["class_argmax"] == nu_class_id)
    iou = pred["iou_vs_gt_nu"]
    # M1
    if is_nu.any():
        nu_idx = np.flatnonzero(is_nu)
        best_in_nu = nu_idx[int(np.argmax(iou[nu_idx]))]
        if iou[best_in_nu] > 0:
            out["has_nu_prediction"] = True
            out["m1_iou"] = float(iou[best_in_nu])
            out["m1_slice_id"] = int(pred["query_id"][best_in_nu])
            out["m3_chi2_nu"] = float(pred["chi2"][best_in_nu, default_idx])
            out["m3_delta_chi2"] = (
                pred["chi2"][best_in_nu, :] - gt_baseline["chi2"][:]
            ).astype(np.float32)
    # M4_all and M4_nu
    # The "GT-best-match slice" is the one with max IoU vs GT-nu across
    # ALL pred slices (no class filtering). If iou == 0 everywhere, no
    # rank computable.
    best_all = int(np.argmax(iou))
    best_iou_all = float(iou[best_all])
    for ti in range(n_th):
        chi2_col = pred["chi2"][:, ti]
        # ALL-pool rank
        if best_iou_all > 0 and not np.isnan(chi2_col[best_all]):
            survive_mask = ~np.isnan(chi2_col)
            survive_idxs = np.flatnonzero(survive_mask)
            # Sort survivors by chi-2 ascending; find position of best_all.
            order = survive_idxs[np.argsort(chi2_col[survive_idxs])]
            pos = int(np.flatnonzero(order == best_all)[0]) + 1  # 1-based
            out["m4_rank_all"][ti] = pos
        # NU-pool rank
        if is_nu.any():
            nu_idxs = np.flatnonzero(is_nu)
            nu_best_local = nu_idxs[int(np.argmax(iou[nu_idxs]))]
            if iou[nu_best_local] > 0 and not np.isnan(chi2_col[nu_best_local]):
                nu_survive = nu_idxs[~np.isnan(chi2_col[nu_idxs])]
                if len(nu_survive) > 0:
                    order = nu_survive[np.argsort(chi2_col[nu_survive])]
                    pos = int(np.flatnonzero(order == nu_best_local)[0]) + 1
                    out["m4_rank_nu"][ti] = pos
    return out


# ---------------------------------------------------------------------------
# GT-baseline (the single in-time-matched nu slice's chi-2)
# ---------------------------------------------------------------------------

def _compute_gt_baseline(
    inf, in_time, gt_nu_mask,
    gamma_by_producer, readout_factor_by_producer,
    photonlib_cache, oob_thresholds, f_sys, eps,
    v_drift, charge_source="raw_adc", share_norm="none",
):
    """Predicted PE + chi-2 sweep for the GT-nu union mask."""
    have_raw = (charge_source == "raw_adc" and "post_rawpix" in inf)
    pos = inf["post_coord"][gt_nu_mask]
    px  = (inf["post_rawpix"] if have_raw else inf["post_pixval"])[gt_nu_mask]
    n_sp = int(gt_nu_mask.sum())
    n_pmts = flash_chi2.N_PMTS
    if n_sp == 0:
        return dict(
            n_sp=0, oob_frac=1.0,
            pe_pred=np.zeros(n_pmts, dtype=np.float32),
            chi2=np.full(len(oob_thresholds), np.nan, dtype=np.float32),
        )
    charge = flash_predict.select_charge_y_with_uv_fallback_np(px)
    # Same per-slice pixel-share normalization as the predicted slices, with
    # the GT-nu union treated as a single cluster — keeps GT baseline and
    # predicted slices apples-to-apples.
    if share_norm == "per-slice" and have_raw:
        share = _pixel_share_counts(
            px, inf["post_tick"][gt_nu_mask],
            inf["post_uwire"][gt_nu_mask], inf["post_ywire"][gt_nu_mask],
            np.zeros(n_sp, dtype=np.int64),
        )
        charge = charge / share.astype(np.float32)
    pe_pred = flash_predict.predict_slice_pe(
        pos_cm=pos, charge=charge,
        flash_t0_us=in_time["t0_us"],
        producer_id=in_time["producer_id"],
        gamma_by_producer=gamma_by_producer,
        readout_factor_by_producer=readout_factor_by_producer,
        photonlib_cache=photonlib_cache,
        v_drift_cm_per_us=v_drift,
    )
    sub = flash_chi2.chi2_with_oob(
        pos_cm=pos, pe_pred=pe_pred, pe_obs=in_time["pe_obs"],
        flash_t0_us=in_time["t0_us"],
        oob_thresholds=oob_thresholds,
        f_sys=f_sys, eps=eps,
    )
    return dict(
        n_sp=n_sp, oob_frac=sub["oob_frac"],
        pe_pred=pe_pred.astype(np.float32),
        chi2=sub["chi2"].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _sentinel_path_for(output_h5):
    """Derive the skip-sentinel path for a would-be perevent output path.
    Replaces the `perevent_` basename prefix with `skipped_` so the
    sentinel is excluded by `aggregate_metrics.py`'s default
    `perevent_*.h5` glob. If the basename doesn't start with `perevent_`,
    inserts `.skipped` before the extension."""
    d, b = os.path.split(output_h5)
    if b.startswith("perevent_"):
        new_b = "skipped_" + b[len("perevent_"):]
    else:
        stem, ext = os.path.splitext(b)
        new_b = stem + ".skipped" + ext
    return os.path.join(d, new_b)


def _write_skip_sentinel(path, run, subrun, event, fileno, entry,
                         model_tag, reason):
    """Write a tiny H5 marker recording that this event was skipped.
    The aggregator's default glob (`perevent_*.h5`) ignores these so
    skipped events don't poison the summary. The driver's skip-if-exists
    check looks for these alongside the real perevent outputs to avoid
    re-attempting events that have already been classified as skips."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["skipped"]  = True
        f.attrs["reason"]   = str(reason)
        f.attrs["run"]      = int(run)
        f.attrs["subrun"]   = int(subrun)
        f.attrs["event"]    = int(event)
        if fileno is not None:
            f.attrs["fileno"] = int(fileno)
        if entry is not None:
            f.attrs["entry"]  = int(entry)
        f.attrs["model_tag"] = str(model_tag)
    os.replace(tmp, path)


def _resolve_output_path(out_path, run, subrun, event, fileno=None, entry=None):
    """If `out_path` points at a directory (or ends in '/'), append a
    canonical filename inside it. When `fileno` and `entry` are both
    given, the filename includes them so the output can be matched 1:1
    with the manifest row:
        perevent_fileno<NNNNN>_entry<NNNNNN>_run<R>_sub<S>_evt<E>.h5
    Otherwise the older `perevent_<run>_<sub>_<evt>.h5` form is used."""
    is_dir_like = (
        out_path.endswith(os.sep)
        or out_path.endswith("/")
        or os.path.isdir(out_path)
    )
    if is_dir_like:
        if fileno is not None and entry is not None:
            canonical = (
                f"perevent_fileno{int(fileno):05d}_entry{int(entry):06d}"
                f"_run{int(run)}_sub{int(subrun)}_evt{int(event)}.h5"
            )
        else:
            canonical = f"perevent_{int(run)}_{int(subrun)}_{int(event)}.h5"
        return os.path.join(out_path.rstrip("/"), canonical)
    return out_path


def _write_perevent_h5(out_path, run, subrun, event, model_tag,
                       oob_thresholds, default_idx,
                       fi, inf, in_time, gt_baseline, pred, metrics,
                       overclaim, charge_source="raw_adc", share_norm="none"):
    """Write the per-event H5 with the documented schema."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["run"]    = int(run)
        f.attrs["subrun"] = int(subrun)
        f.attrs["event"]  = int(event)
        f.attrs["model_tag"] = str(model_tag)
        f.attrs["has_nu_prediction"] = bool(metrics["has_nu_prediction"])
        f.attrs["oob_thresholds"] = oob_thresholds.astype(np.float32)
        f.attrs["default_oob_idx"] = int(default_idx)
        f.attrs["category_names"] = np.array(
            categorize.CATEGORY_NAMES, dtype="S32",
        )
        f.attrs["nu_class_id"]        = 0
        f.attrs["no_object_class_id"] = int(inf["no_object_class_id"])
        f.attrs["charge_source"]      = str(charge_source)
        f.attrs["charge_share_norm"]  = str(share_norm)

        # truth/
        tr = f.create_group("truth")
        tr.attrs["category_mask"]        = int(fi["category_mask"])
        tr.attrs["n_visible_nu_gammas"]  = int(fi["n_visible_nu_gammas"])
        tr.attrs["n_primary_pi0"]        = int(fi["n_primary_pi0"])
        for k, v in fi["truth"].items():
            tr.attrs[k] = v

        # in_time_flash/
        it = f.create_group("in_time_flash")
        it.attrs["flash_idx"]       = int(in_time["flash_idx"])
        it.attrs["t0_us"]           = float(in_time["t0_us"])
        it.attrs["producer_id"]     = int(in_time["producer_id"])
        it.attrs["paired_slice_id"] = int(in_time["paired_slice_id"])
        it.create_dataset("pe_obs", data=in_time["pe_obs"].astype(np.float32))

        # gt_baseline/
        gb = f.create_group("gt_baseline")
        gb.attrs["n_sp"]    = int(gt_baseline["n_sp"])
        gb.attrs["oob_frac"] = float(gt_baseline["oob_frac"])
        gb.create_dataset("pe_pred", data=gt_baseline["pe_pred"])
        gb.create_dataset("chi2",    data=gt_baseline["chi2"])

        # pred_slices/
        ps = f.create_group("pred_slices")
        ps.attrs["n_pred_slices"]    = int(pred["query_id"].shape[0])
        ps.attrs["n_pred_nu_slices"] = int((pred["class_argmax"] == 0).sum())
        ps.create_dataset("query_id",     data=pred["query_id"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("class_argmax", data=pred["class_argmax"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("n_sp",         data=pred["n_sp"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("iou_vs_gt_nu", data=pred["iou_vs_gt_nu"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("oob_frac",     data=pred["oob_frac"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("chi2",         data=pred["chi2"],
                          compression="gzip", compression_opts=4)
        ps.create_dataset("pe_pred",      data=pred["pe_pred"],
                          compression="gzip", compression_opts=4)

        # metrics/
        m = f.create_group("metrics")
        m.attrs["has_nu_prediction"] = bool(metrics["has_nu_prediction"])
        m.attrs["m1_iou"]            = float(metrics["m1_iou"])
        m.attrs["m1_slice_id"]       = int(metrics["m1_slice_id"])
        # Per-pair "intent" view of M1 — see overclaim/ below. Surfaces
        # when the model's class-correct nu query loses the panoptic
        # argmax to over-claiming competitors.
        m.attrs["m1_iou_intent"]      = float(overclaim["m1_iou_intent"])
        m.attrs["m1_slice_id_intent"] = int(overclaim["m1_slice_id_intent"])
        m.attrs["m3_chi2_gt"]        = float(metrics["m3_chi2_gt"])
        m.attrs["m3_chi2_nu"]        = float(metrics["m3_chi2_nu"])
        m.create_dataset("m3_delta_chi2", data=metrics["m3_delta_chi2"])
        m.create_dataset("m4_rank_all",   data=metrics["m4_rank_all"])
        m.create_dataset("m4_rank_nu",    data=metrics["m4_rank_nu"])

        # overclaim/  — the panoptic-vs-per-pair gap diagnostic
        oc = f.create_group("overclaim")
        for k, key in (
            ("sp_level_nu_recall",            "sp_level_nu_recall"),
            ("sp_level_nu_precision",         "sp_level_nu_precision"),
        ):
            oc.attrs[k] = float(overclaim[key])
        for k in ("n_post_sp_total", "n_post_sp_gt_nu",
                  "n_post_sp_pred_nu", "n_post_sp_true_and_pred_nu",
                  "n_nu_gt_instances",
                  "n_nu_gt_matched_to_nu_query",
                  "n_nu_match_survived_panoptic"):
            oc.attrs[k] = int(overclaim[k])
        for k in ("gt_nu_idx", "matched_query_id", "matched_query_class",
                  "matched_query_pair_iou", "matched_query_argmax_iou",
                  "matched_query_panoptic_n_sp"):
            oc.create_dataset(k, data=overclaim[k],
                              compression="gzip", compression_opts=4)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--merged-h5",    required=True)
    ap.add_argument("--flashinfo-h5", required=True)
    ap.add_argument("--inference-h5", required=True)
    ap.add_argument("--output-h5",    required=True)
    ap.add_argument("--model-tag", default="unknown",
                    help="String tag attached to the per-event H5.")
    ap.add_argument("--fileno", type=int, default=None,
                    help="Optional manifest fileno; when given with --entry, "
                         "the auto-derived output filename includes both so "
                         "the perevent H5 maps 1:1 to its manifest row.")
    ap.add_argument("--entry", type=int, default=None,
                    help="Optional manifest entry; see --fileno.")

    # PhotonLib + γ + readout-factor.
    ap.add_argument("--photonlib-cache", default=None,
                    help=f"Default: {flash_predict.PHOTONLIB_DEFAULT_CACHE}")
    ap.add_argument("--gamma-beam",   type=float, default=flash_predict.DEFAULT_GAMMA)
    ap.add_argument("--gamma-cosmic", type=float, default=flash_predict.DEFAULT_GAMMA)
    ap.add_argument("--readout-fast-fraction",  type=float,
                    default=flash_predict.DEFAULT_FAST_FRACTION)
    ap.add_argument("--readout-tau-slow-us",    type=float,
                    default=flash_predict.DEFAULT_TAU_SLOW_US)
    ap.add_argument("--readout-window-cosmic-us", type=float,
                    default=flash_predict.DEFAULT_WINDOW_COSMIC_US)
    ap.add_argument("--readout-factor-beam",    type=float,
                    default=flash_predict.DEFAULT_READOUT_FACTOR_BEAM)
    ap.add_argument("--v-drift-cm-per-us", type=float,
                    default=flash_predict.DEFAULT_V_DRIFT_CM_PER_US)

    # Charge source + pixel-share normalization.
    ap.add_argument("--charge-source", choices=["raw_adc", "pre_pixval"],
                    default="raw_adc",
                    help="Per-SP charge for the flash prediction. 'raw_adc' "
                         "(default) joins the merged H5 for the linear Y-plane "
                         "ADC; 'pre_pixval' uses the inference H5's "
                         "log-normalized backbone feature (legacy, back-compat).")
    ap.add_argument("--charge-share-norm", choices=["none", "per-slice"],
                    default="per-slice",
                    help="Divide each SP's charge by the number of SPs in the "
                         "same slice sharing its source wire pixel (conserves "
                         "the 2D pixel-footprint integral; mitigates "
                         "ghost-driven over-counting). Requires "
                         "--charge-source raw_adc. Pass 'none' to reproduce "
                         "the legacy (no-share) behavior of existing perevent "
                         "sets.")

    # Chi-2 + OOB knobs.
    ap.add_argument("--f-sys", type=float, default=0.10)
    ap.add_argument("--eps",   type=float, default=1.0)
    ap.add_argument("--oob-thresholds", type=str,
                    default=",".join(str(x) for x in DEFAULT_OOB_THRESHOLDS),
                    help=f"Comma-separated OOB thresholds. Default: "
                         f"{','.join(str(x) for x in DEFAULT_OOB_THRESHOLDS)}")
    ap.add_argument("--default-oob-idx", type=int, default=DEFAULT_OOB_IDX,
                    help=f"Index into --oob-thresholds used as the M3 "
                         f"headline. Default: {DEFAULT_OOB_IDX}")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    oob_thresholds = np.array(
        [float(x) for x in args.oob_thresholds.split(",")], dtype=np.float32,
    )
    if args.default_oob_idx >= len(oob_thresholds):
        ap.error(f"--default-oob-idx {args.default_oob_idx} >= "
                 f"len(oob_thresholds)={len(oob_thresholds)}")

    gamma_by_producer = (args.gamma_beam, args.gamma_cosmic)
    readout_factor_by_producer = (
        args.readout_factor_beam,
        flash_predict.compute_readout_factor(
            args.readout_fast_fraction,
            args.readout_tau_slow_us,
            args.readout_window_cosmic_us,
        ),
    )

    if args.verbose:
        print(f"merged_h5:    {args.merged_h5}")
        print(f"flashinfo_h5: {args.flashinfo_h5}")
        print(f"inference_h5: {args.inference_h5}")
        print(f"output_h5:    {args.output_h5}")
        print(f"γ_beam={gamma_by_producer[0]:.4g}, "
              f"γ_cosmic={gamma_by_producer[1]:.4g}")
        print(f"readout_factor (beam, cosmic) = "
              f"({readout_factor_by_producer[0]:.4f}, "
              f"{readout_factor_by_producer[1]:.4f})")
        print(f"oob_thresholds = {oob_thresholds}")
        print(f"default_oob_idx = {args.default_oob_idx}  "
              f"(= {float(oob_thresholds[args.default_oob_idx]):.2f})")

    # Load.
    inf = _load_inference(args.inference_h5)
    fi  = _load_flashinfo(args.flashinfo_h5)
    run, subrun, event = inf["run"], inf["subrun"], inf["event"]

    # Allow --output-h5 to be a directory; if so, fill in the canonical
    # filename inside it (includes fileno/entry when both are provided).
    args.output_h5 = _resolve_output_path(
        args.output_h5, run, subrun, event,
        fileno=args.fileno, entry=args.entry,
    )
    if args.verbose:
        print(f"resolved output_h5: {args.output_h5}")
    if (run, subrun, event) != (fi["run"], fi["subrun"], fi["event"]):
        sys.stderr.write(
            f"[warn] (run,subrun,event) mismatch:\n"
            f"  inference: ({run},{subrun},{event})\n"
            f"  flashinfo: ({fi['run']},{fi['subrun']},{fi['event']})\n"
        )
    if args.verbose:
        print(f"(run, subrun, event) = ({run}, {subrun}, {event})  "
              f"category = {categorize.category_str(fi['category_mask'])}")

    # Identify in-time flash + GT-nu union mask.
    in_time = _find_in_time_flash(fi)
    if in_time is None:
        sentinel = _sentinel_path_for(args.output_h5)
        _write_skip_sentinel(
            sentinel,
            run=run, subrun=subrun, event=event,
            fileno=args.fileno, entry=args.entry,
            model_tag=args.model_tag,
            reason="no_flashes_in_flashinfo",
        )
        sys.stderr.write(
            f"event ({run},{subrun},{event}): no flashes at all in "
            f"flashinfo — cannot compute chi-2; wrote skip sentinel "
            f"{sentinel} and exiting 0\n"
        )
        return
    if args.verbose:
        print(f"in-time flash idx={in_time['flash_idx']} "
              f"t0={in_time['t0_us']:.3f}us "
              f"producer={in_time['producer_id']} "
              f"totalPE={float(in_time['pe_obs'].sum()):.1f} "
              f"paired_slice_id={in_time['paired_slice_id']}")

    gt_nu_mask = _gt_nu_sp_mask(inf)
    if args.verbose:
        print(f"GT-nu mask: {int(gt_nu_mask.sum())} / "
              f"{len(gt_nu_mask)} post-SPs")

    # Recover raw-ADC charge + wire/tick per post-SP from the merged H5.
    charge_source = args.charge_source
    share_norm = args.charge_share_norm
    if charge_source == "raw_adc":
        mp = _load_merged_perpoint(args.merged_h5, inf)
        if mp is None:
            sys.stderr.write(
                "[warn] merged-H5 coordinate join failed (unmatched SPs); "
                "falling back to log-normalized pre/pixval charge and "
                "disabling pixel-share normalization\n"
            )
            charge_source = "pre_pixval"
            share_norm = "none"
        else:
            inf.update(mp)
    elif share_norm == "per-slice":
        sys.stderr.write(
            "[warn] --charge-share-norm per-slice requires "
            "--charge-source raw_adc; disabling share normalization\n"
        )
        share_norm = "none"
    if args.verbose:
        print(f"charge_source={charge_source}  share_norm={share_norm}")

    # GT-baseline.
    gt_baseline = _compute_gt_baseline(
        inf, in_time, gt_nu_mask,
        gamma_by_producer, readout_factor_by_producer,
        args.photonlib_cache, oob_thresholds, args.f_sys, args.eps,
        args.v_drift_cm_per_us, charge_source, share_norm,
    )

    # Per-pred-slice.
    pred = _compute_pred_slices(
        inf, in_time, gt_nu_mask,
        gamma_by_producer, readout_factor_by_producer,
        args.photonlib_cache, oob_thresholds, args.f_sys, args.eps,
        args.v_drift_cm_per_us, charge_source, share_norm,
    )
    if args.verbose:
        print(f"n_pred_slices={pred['query_id'].shape[0]}  "
              f"n_pred_nu={int((pred['class_argmax'] == 0).sum())}  "
              f"max_iou={float(pred['iou_vs_gt_nu'].max() if pred['iou_vs_gt_nu'].size else 0):.3f}")

    # Metrics.
    metrics = _compute_metrics(
        pred, gt_baseline, oob_thresholds, args.default_oob_idx,
        no_object_class_id=inf["no_object_class_id"], nu_class_id=0,
    )
    # Over-claim metrics (per-pair view + sp-level confusion). Computed
    # from the inference H5 directly — independent of the panoptic
    # slice iteration. See `_compute_overclaim` docstring.
    overclaim = _compute_overclaim(inf, nu_class_id=0)
    if args.verbose:
        print(f"M1 IoU panoptic = {metrics['m1_iou']:.3f}  "
              f"(slice_id={metrics['m1_slice_id']}, "
              f"has_nu_pred={metrics['has_nu_prediction']})")
        print(f"M1 IoU intent   = {overclaim['m1_iou_intent']:.3f}  "
              f"(slice_id={overclaim['m1_slice_id_intent']})")
        print(f"M3 chi2_gt={metrics['m3_chi2_gt']:.2f}  "
              f"chi2_nu={metrics['m3_chi2_nu']:.2f}  "
              f"delta_chi2[default]={float(metrics['m3_delta_chi2'][args.default_oob_idx]):.2f}")
        print(f"M4 rank_all={metrics['m4_rank_all']}  "
              f"rank_nu={metrics['m4_rank_nu']}")
        print(f"overclaim: sp_recall={overclaim['sp_level_nu_recall']:.3f}  "
              f"sp_precision={overclaim['sp_level_nu_precision']:.3f}  "
              f"n_nu_gt={overclaim['n_nu_gt_instances']}  "
              f"n_matched_to_nu={overclaim['n_nu_gt_matched_to_nu_query']}  "
              f"n_survived_panoptic={overclaim['n_nu_match_survived_panoptic']}")

    # Write.
    _write_perevent_h5(
        args.output_h5, run, subrun, event, args.model_tag,
        oob_thresholds, args.default_oob_idx,
        fi, inf, in_time, gt_baseline, pred, metrics, overclaim,
        charge_source, share_norm,
    )
    if args.verbose:
        print(f"wrote {args.output_h5}")


if __name__ == "__main__":
    main()
