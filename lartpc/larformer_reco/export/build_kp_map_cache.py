#!/usr/bin/env python3
"""Prebuild the src_file -> (gidx, path) cache of one or more keypoint2 lists
(`<list>.srcmap.json`, see export_gen2ntuple.build_kp_map) so that export
shards do not each open every kp2 file of the list (176k Lustre opens per
shard for bnb5e19). Run inside the container after the regen step:

  python3 lartpc/larformer_reco/export/build_kp_map_cache.py <kp2 list> [<kp2 list> ...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from lartpc.larformer_reco.export.export_gen2ntuple import build_kp_map, _kp_cache_path  # noqa: E402
from lartpc.larformer_reco.utils import read_list  # noqa: E402

for lp in sys.argv[1:]:
    t0 = time.time()
    m = build_kp_map(read_list(lp), list_path=lp)
    print(f">>> {lp}: {len(m)} entries -> {_kp_cache_path(lp)} ({time.time() - t0:.1f} s)")
