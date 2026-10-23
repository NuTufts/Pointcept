# Stage-B figures: prod

| run | nodes | proc/GPU | MPS | events | ev/shard | wall s | ev/s | ev/GPU-h (wall) | ev/GPU-h (inference) | s/ev per proc | GPU util % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| testB_p1m0 | 1 | 1 | 0 | 160 | 40 | 120 | 1.33 | 1200 | 2105 | 1.71 | 21.1 |
| testB_p2m1 | 1 | 2 | 1 | 320 | 40 | 133 | 2.41 | 2165 | 3731 | 1.93 | 41.3 |
| testB_p3m1 | 1 | 3 | 1 | 480 | 40 | 248 | 1.94 | 1742 | 4977 | 2.17 | 32.9 |
| prod | 10 | 2 | 1 | 87383 | 1092 | 2367 | 36.92 | 3323 | 3600 | 2.00 | 57.2 |

Production phases (median over 80 shards): model build 74 s, inference 36.5 min, pack + copy 2 s; mean utilization during the inference phase 59.9 % (active-window mean incl. build/tail 57.2 %).
Per-node utilization range over the whole job: 55-58 %.

figures: /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/polaris_logs/figures/stageB_gpu_util_timeseries.png, /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/polaris_logs/figures/stageB_config_matrix.png
