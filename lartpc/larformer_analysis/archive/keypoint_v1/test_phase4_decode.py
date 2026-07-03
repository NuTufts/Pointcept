"""Phase 4 unit test: keypoint decode + PPN-style metrics on synthetic data.

Validates the decode/match math in isolation (no model). Builds a synthetic
event with SP clusters around two known keypoints, gives them high scores +
offsets pointing at the keypoints, and checks that:
  - decode_dense_votes recovers the keypoints (clustered votes near GT),
  - accumulate_metrics reports recall 1.0 within threshold,
  - decode_query_points emits start/end for a real query and skips no_object.

    python lartpc/larformer_analysis/archive/keypoint_v1/test_phase4_decode.py
"""

import sys

import numpy as np

from pointcept.models.LArFormer.keypoint_eval import (
    decode_dense_votes, decode_query_points, accumulate_metrics,
    format_metrics_table, denorm, query_points_to_typed, reconcile_keypoints,
)
from lartpc.data_prep.labels.keypoint_labels import (
    KPTYPE_TRACK_START, KPTYPE_TRACK_END, KPTYPE_SHOWER, KPTYPE_NU_VERTEX,
)

CENTER = np.array([125.0, 0.0, 518.0], dtype=np.float32)
SCALE = 179.55
THR = [3.0, 10.0]


def main():
    rng = np.random.RandomState(0)

    # Two GT keypoints of type 1 (track_start), in cm, placed BETWEEN points.
    gt_cm = np.array([[130.0, 5.0, 520.0], [100.0, -10.0, 600.0]],
                     dtype=np.float32)
    gt_norm = (gt_cm - CENTER) / SCALE

    # 40 SPs scattered ~2 cm around each GT (so no SP is exactly on it).
    pts = []
    for g in gt_norm:
        pts.append(g + rng.randn(40, 3).astype(np.float32) * (2.0 / SCALE))
    coord_norm = np.concatenate(pts + [rng.randn(50, 3).astype(np.float32)*0.3],
                                axis=0)            # +50 background SPs
    N = coord_norm.shape[0]
    n_types = 6
    scores = np.zeros((N, n_types), dtype=np.float32)
    offsets = np.zeros((N, n_types, 3), dtype=np.float32)
    # First 80 SPs vote for their nearest GT (type 1) with high score.
    for i in range(80):
        g = gt_norm[0] if i < 40 else gt_norm[1]
        scores[i, 1] = 0.9
        offsets[i, 1, :] = g - coord_norm[i]       # exact vote to the GT

    decoded = decode_dense_votes(
        coord_norm, scores, offsets, CENTER, SCALE,
        score_thresh=0.3, cluster_radius_cm=3.0, n_types=n_types)
    clusters = decoded[1]
    print(f"[p4] type=track_start decoded {len(clusters)} cluster(s)")
    for c in clusters:
        d = np.linalg.norm(gt_cm - c["pos_cm"], axis=1).min()
        print(f"     pos={np.round(c['pos_cm'],2)} score={c['score']:.2f} "
              f"n={c['n_votes']}  d_to_GT={d:.3f}cm")
    assert len(clusters) == 2, f"expected 2 clusters, got {len(clusters)}"
    for c in clusters:
        assert np.linalg.norm(gt_cm - c["pos_cm"], axis=1).min() < 1.0

    # Metrics: build per-type pred/gt dicts and accumulate.
    pred_by_type = {t: decoded.get(t, []) for t in range(n_types)}
    gt_by_type = {1: gt_cm}
    metrics = accumulate_metrics([pred_by_type], [gt_by_type], THR, n_types)
    print("\n" + format_metrics_table(metrics, THR, n_types))
    assert metrics[1][3.0]["recall"] == 1.0
    assert metrics[1][3.0]["precision"] == 1.0
    assert metrics[1][3.0]["tp"] == 2

    # Background-only type → no clusters, recall stays 0 with no GT (no crash).
    assert metrics[0][3.0]["tp"] == 0 and metrics[0][3.0]["fn"] == 0

    # ---- query decode ----
    C = 8  # 7 classes + no_object(=7)
    class_logits = np.full((3, C), -5.0, dtype=np.float32)
    class_logits[0, 2] = 5.0     # query 0 → mu (track) high prob
    class_logits[1, 1] = 5.0     # query 1 → gamma (shower) high prob
    class_logits[2, 7] = 5.0     # query 2 → no_object → skipped
    origin_norm = np.zeros((3, 3), dtype=np.float32)
    end_norm = np.ones((3, 3), dtype=np.float32) * 0.1
    end_logit = np.array([5.0, -5.0, 5.0], dtype=np.float32)  # q0 has end
    qpts = decode_query_points(
        class_logits, origin_norm, CENTER, SCALE,
        end_norm=end_norm, end_logit=end_logit, no_object_class_id=7,
        class_prob_floor=0.3, end_prob_floor=0.5, track_class_ids={2, 3, 4, 5})
    kinds = [(p["query"], p["kind"], p["class_id"]) for p in qpts]
    print(f"\n[p4] query decode → {kinds}")
    # q0: start+end (track, end gate on); q1: start only (shower); q2: none
    assert (0, "start", 2) in kinds and (0, "end", 2) in kinds
    assert (1, "start", 1) in kinds and not any(k[0] == 1 and k[1] == "end" for k in kinds)
    assert not any(k[0] == 2 for k in kinds)

    # ---- query → typed mapping + reconcile ----
    typed = query_points_to_typed(qpts)   # qpts from the query decode above
    # q0 mu(track): start→track_start, end→track_end; q1 gamma(shower): start→shower
    assert len(typed[KPTYPE_TRACK_START]) == 1
    assert len(typed[KPTYPE_TRACK_END]) == 1
    assert len(typed[KPTYPE_SHOWER]) == 1
    print(f"[p4] query→typed: track_start={len(typed[KPTYPE_TRACK_START])} "
          f"track_end={len(typed[KPTYPE_TRACK_END])} "
          f"shower={len(typed[KPTYPE_SHOWER])}")

    # reconcile: vertex from dense (the synthetic dense `decoded` has type-1
    # clusters but no type-0); query types take the query points.
    final = reconcile_keypoints(decoded, typed)
    assert final[KPTYPE_TRACK_START] is typed[KPTYPE_TRACK_START]  # query wins
    assert final[KPTYPE_NU_VERTEX] == decoded.get(KPTYPE_NU_VERTEX, [])  # dense
    # Fallback: empty query type → dense. decoded has track_start clusters;
    # an empty query_typed for that type should fall back to them.
    empty_typed = {t: [] for t in range(6)}
    fb = reconcile_keypoints(decoded, empty_typed)
    assert fb[KPTYPE_TRACK_START] == decoded[KPTYPE_TRACK_START]
    print("[p4] reconcile: query types prefer query points, fall back to dense")

    print("\n[p4] PASS: dense decode recovers keypoints, metrics correct, "
          "query decode gates start/end, typed-map + reconcile correct")
    sys.exit(0)


if __name__ == "__main__":
    main()
