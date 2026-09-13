#!/usr/bin/env python3
"""Post-hadd assertion (PyROOT, inside the container after thisroot.sh):
merged EventTree entries == sum over shards; reports potTree entries and the
showerCosmicScore version fingerprint.  usage: hadd_check.py <merged.root> <shard.root>..."""
import sys

import ROOT

out, shards = sys.argv[1], sys.argv[2:]
tot = 0
for s in shards:
    f = ROOT.TFile.Open(s)
    tot += f.Get("EventTree").GetEntries()
    f.Close()
f = ROOT.TFile.Open(out)
n = f.Get("EventTree").GetEntries()
npot = f.Get("potTree").GetEntries() if f.Get("potTree") else -1
has_score = bool(f.Get("EventTree").GetBranch("showerCosmicScore"))
f.Close()
assert n == tot, f"entry mismatch: merged {n} != shard sum {tot}"
print(f">>> merged EventTree entries: {n} (== shard sum over {len(shards)} shards), "
      f"potTree: {npot}, showerCosmicScore branch: {has_score}")
