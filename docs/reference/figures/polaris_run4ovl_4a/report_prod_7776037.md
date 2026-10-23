# Stage-B GPU pass report: prod

nodes 10 x 4 GPU, processes per GPU 2, MPS on, shards 80 (80 done), events 87383

| quantity | value |
|---|---|
| job wall (charged) | 2367 s = 0.66 h |
| node-hours | 6.58 |
| GPU-hours | 26.30 |
| aggregate throughput | 36.917 events/s |
| events per GPU-hour | 3323 |
| events per node-hour | 13290 |
| per-process inference rate (median over shards, excl. model build) | 2.00 s/event |
| model build per shard (median) | 74 s |
| pack + copy per shard (median) | 2 s |
| mean GPU utilization (nvidia-smi, active windows) | 57.2 % over 40 GPUs |
| tracebacks | 0 |

## Per node

| node | shards | events | active s | GPU util % (0..3) | max mem MiB | mean W |
|---|---|---|---|---|---|---|
| x3102c0s13b0n0 | 8 | 8744 | 2331 | 56 / 60 / 55 / 54 | 36658 / 39228 / 37070 / 39692 | 180 / 189 / 163 / 168 |
| x3102c0s13b1n0 | 8 | 8744 | 2288 | 55 / 57 / 56 / 60 | 38984 / 38882 / 36176 / 40148 | 177 / 187 / 166 / 191 |
| x3102c0s19b0n0 | 8 | 8744 | 2342 | 58 / 56 / 54 / 56 | 39210 / 34982 / 39766 / 38290 | 167 / 172 / 171 / 179 |
| x3102c0s19b1n0 | 8 | 8744 | 2363 | 55 / 61 / 55 / 59 | 36040 / 39836 / 40094 / 36956 | 176 / 182 / 179 / 205 |
| x3102c0s1b0n0 | 8 | 8744 | 2287 | 59 / 60 / 56 / 59 | 38702 / 39310 / 39600 / 37262 | 167 / 174 / 168 / 182 |
| x3102c0s1b1n0 | 8 | 8744 | 2291 | 60 / 59 / 56 / 58 | 39020 / 38900 / 36992 / 38060 | 182 / 180 / 171 / 183 |
| x3102c0s25b0n0 | 8 | 8744 | 2329 | 56 / 57 / 58 / 58 | 37516 / 39042 / 39226 / 38298 | 169 / 181 / 162 / 182 |
| x3102c0s25b1n0 | 8 | 8744 | 2298 | 55 / 56 / 60 / 57 | 39790 / 38772 / 40330 / 32510 | 167 / 188 / 190 / 192 |
| x3102c0s31b0n0 | 8 | 8744 | 2302 | 58 / 57 / 58 / 59 | 40278 / 39284 / 39598 / 38690 | 176 / 184 / 173 / 186 |
| x3102c0s31b1n0 | 8 | 8687 | 2326 | 55 / 53 / 58 / 56 | 35962 / 39354 / 40182 / 36962 | 170 / 174 / 168 / 200 |

## Per shard

| shard | host | gpu | events | build s | s/event | pack s | done |
|---|---|---|---|---|---|---|---|
| shard000 | x3102c0s13b0n0 | 0 | 1093 | 74 | 1.99 | 2 | yes |
| shard001 | x3102c0s13b1n0 | 0 | 1093 | 74 | 1.99 | 2 | yes |
| shard002 | x3102c0s19b0n0 | 0 | 1093 | 75 | 2.05 | 1 | yes |
| shard003 | x3102c0s19b1n0 | 0 | 1093 | 74 | 2.01 | 2 | yes |
| shard004 | x3102c0s1b0n0 | 0 | 1093 | 75 | 2.01 | 2 | yes |
| shard005 | x3102c0s1b1n0 | 0 | 1093 | 74 | 2.00 | 2 | yes |
| shard006 | x3102c0s25b0n0 | 0 | 1093 | 75 | 2.01 | 1 | yes |
| shard007 | x3102c0s25b1n0 | 0 | 1093 | 76 | 2.01 | 1 | yes |
| shard008 | x3102c0s31b0n0 | 0 | 1093 | 74 | 1.98 | 2 | yes |
| shard009 | x3102c0s31b1n0 | 0 | 1093 | 73 | 2.01 | 1 | yes |
| shard010 | x3102c0s13b0n0 | 1 | 1093 | 75 | 2.01 | 1 | yes |
| shard011 | x3102c0s13b1n0 | 1 | 1093 | 74 | 2.01 | 1 | yes |
| shard012 | x3102c0s19b0n0 | 1 | 1093 | 75 | 2.03 | 2 | yes |
| shard013 | x3102c0s19b1n0 | 1 | 1093 | 75 | 2.02 | 1 | yes |
| shard014 | x3102c0s1b0n0 | 1 | 1093 | 75 | 2.00 | 2 | yes |
| shard015 | x3102c0s1b1n0 | 1 | 1093 | 75 | 2.00 | 2 | yes |
| shard016 | x3102c0s25b0n0 | 1 | 1093 | 75 | 2.06 | 1 | yes |
| shard017 | x3102c0s25b1n0 | 1 | 1093 | 74 | 2.02 | 2 | yes |
| shard018 | x3102c0s31b0n0 | 1 | 1093 | 74 | 1.96 | 1 | yes |
| shard019 | x3102c0s31b1n0 | 1 | 1093 | 73 | 1.80 | 1 | yes |
| shard020 | x3102c0s13b0n0 | 2 | 1093 | 73 | 1.80 | 2 | yes |
| shard021 | x3102c0s13b1n0 | 2 | 1093 | 74 | 1.90 | 1 | yes |
| shard022 | x3102c0s19b0n0 | 2 | 1093 | 74 | 1.87 | 1 | yes |
| shard023 | x3102c0s19b1n0 | 2 | 1093 | 74 | 1.96 | 2 | yes |
| shard024 | x3102c0s1b0n0 | 2 | 1093 | 75 | 1.97 | 1 | yes |
| shard025 | x3102c0s1b1n0 | 2 | 1093 | 75 | 1.92 | 2 | yes |
| shard026 | x3102c0s25b0n0 | 2 | 1093 | 74 | 1.91 | 1 | yes |
| shard027 | x3102c0s25b1n0 | 2 | 1093 | 74 | 1.95 | 2 | yes |
| shard028 | x3102c0s31b0n0 | 2 | 1093 | 73 | 1.92 | 1 | yes |
| shard029 | x3102c0s31b1n0 | 2 | 1093 | 74 | 1.98 | 2 | yes |
| shard030 | x3102c0s13b0n0 | 3 | 1093 | 74 | 2.00 | 2 | yes |
| shard031 | x3102c0s13b1n0 | 3 | 1093 | 75 | 1.95 | 1 | yes |
| shard032 | x3102c0s19b0n0 | 3 | 1093 | 75 | 1.97 | 2 | yes |
| shard033 | x3102c0s19b1n0 | 3 | 1093 | 74 | 2.02 | 2 | yes |
| shard034 | x3102c0s1b0n0 | 3 | 1093 | 74 | 2.00 | 2 | yes |
| shard035 | x3102c0s1b1n0 | 3 | 1093 | 75 | 2.03 | 2 | yes |
| shard036 | x3102c0s25b0n0 | 3 | 1093 | 75 | 1.96 | 1 | yes |
| shard037 | x3102c0s25b1n0 | 3 | 1093 | 75 | 2.02 | 2 | yes |
| shard038 | x3102c0s31b0n0 | 3 | 1093 | 74 | 2.01 | 2 | yes |
| shard039 | x3102c0s31b1n0 | 3 | 1093 | 74 | 2.06 | 1 | yes |
| shard040 | x3102c0s13b0n0 | 0 | 1093 | 74 | 1.98 | 2 | yes |
| shard041 | x3102c0s13b1n0 | 0 | 1093 | 74 | 1.98 | 1 | yes |
| shard042 | x3102c0s19b0n0 | 0 | 1093 | 75 | 1.98 | 1 | yes |
| shard043 | x3102c0s19b1n0 | 0 | 1093 | 76 | 1.95 | 1 | yes |
| shard044 | x3102c0s1b0n0 | 0 | 1093 | 74 | 1.93 | 1 | yes |
| shard045 | x3102c0s1b1n0 | 0 | 1093 | 73 | 2.03 | 1 | yes |
| shard046 | x3102c0s25b0n0 | 0 | 1093 | 75 | 1.99 | 2 | yes |
| shard047 | x3102c0s25b1n0 | 0 | 1093 | 75 | 1.97 | 2 | yes |
| shard048 | x3102c0s31b0n0 | 0 | 1093 | 74 | 1.99 | 2 | yes |
| shard049 | x3102c0s31b1n0 | 0 | 1093 | 74 | 1.97 | 1 | yes |
| shard050 | x3102c0s13b0n0 | 1 | 1093 | 75 | 2.06 | 1 | yes |
| shard051 | x3102c0s13b1n0 | 1 | 1093 | 74 | 2.02 | 1 | yes |
| shard052 | x3102c0s19b0n0 | 1 | 1093 | 75 | 1.99 | 2 | yes |
| shard053 | x3102c0s19b1n0 | 1 | 1093 | 75 | 2.04 | 1 | yes |
| shard054 | x3102c0s1b0n0 | 1 | 1093 | 73 | 2.02 | 2 | yes |
| shard055 | x3102c0s1b1n0 | 1 | 1093 | 75 | 2.01 | 2 | yes |
| shard056 | x3102c0s25b0n0 | 1 | 1093 | 75 | 2.01 | 2 | yes |
| shard057 | x3102c0s25b1n0 | 1 | 1093 | 74 | 2.01 | 2 | yes |
| shard058 | x3102c0s31b0n0 | 1 | 1093 | 74 | 2.04 | 1 | yes |
| shard059 | x3102c0s31b1n0 | 1 | 1093 | 74 | 1.99 | 2 | yes |
| shard060 | x3102c0s13b0n0 | 2 | 1093 | 74 | 1.97 | 2 | yes |
| shard061 | x3102c0s13b1n0 | 2 | 1093 | 74 | 2.01 | 1 | yes |
| shard062 | x3102c0s19b0n0 | 2 | 1093 | 74 | 1.98 | 2 | yes |
| shard063 | x3102c0s19b1n0 | 2 | 1093 | 74 | 2.01 | 2 | yes |
| shard064 | x3102c0s1b0n0 | 2 | 1093 | 73 | 2.02 | 1 | yes |
| shard065 | x3102c0s1b1n0 | 2 | 1093 | 76 | 1.97 | 2 | yes |
| shard066 | x3102c0s25b0n0 | 2 | 1093 | 75 | 1.98 | 1 | yes |
| shard067 | x3102c0s25b1n0 | 2 | 1093 | 74 | 2.01 | 1 | yes |
| shard068 | x3102c0s31b0n0 | 2 | 1093 | 74 | 2.02 | 2 | yes |
| shard069 | x3102c0s31b1n0 | 2 | 1093 | 76 | 2.04 | 1 | yes |
| shard070 | x3102c0s13b0n0 | 3 | 1093 | 74 | 2.00 | 2 | yes |
| shard071 | x3102c0s13b1n0 | 3 | 1093 | 75 | 2.02 | 1 | yes |
| shard072 | x3102c0s19b0n0 | 3 | 1093 | 74 | 2.07 | 2 | yes |
| shard073 | x3102c0s19b1n0 | 3 | 1093 | 74 | 2.09 | 1 | yes |
| shard074 | x3102c0s1b0n0 | 3 | 1093 | 74 | 2.02 | 2 | yes |
| shard075 | x3102c0s1b1n0 | 3 | 1093 | 75 | 2.01 | 1 | yes |
| shard076 | x3102c0s25b0n0 | 3 | 1093 | 75 | 1.99 | 1 | yes |
| shard077 | x3102c0s25b1n0 | 3 | 1093 | 76 | 2.03 | 1 | yes |
| shard078 | x3102c0s31b0n0 | 3 | 1093 | 74 | 2.00 | 2 | yes |
| shard079 | x3102c0s31b1n0 | 3 | 1036 | 74 | 2.05 | 2 | yes |

```
tag=prod nodes=10 ppg=2 mps=1 shards=80 done=80 events=87383 elapsed=2367s agg=36.917 ev/s per_proc_median=2.00 s/ev gpu_util_mean=57.2% ev_per_gpu_h=3323 ev_per_node_h=13290 node_hours=6.58
```
