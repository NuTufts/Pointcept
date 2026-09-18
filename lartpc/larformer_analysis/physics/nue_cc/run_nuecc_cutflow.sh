#!/bin/bash
# nue-CC cutflow driver -- edit the cut values / step lines below and rerun.
#
#   bash run_nuecc_cutflow.sh                  # run-1 tables (default)
#   TABLES=tables bash run_nuecc_cutflow.sh    # run-3 cew6 tables
#   VARPLOTS=0 bash run_nuecc_cutflow.sh       # numbers + headline plots only (fast)
#   ELCONF=9 PRIM=6 bash run_nuecc_cutflow.sh  # try other cut values without editing
#   EXTRESCALE=1.47 bash run_nuecc_cutflow.sh   # scale the EXT prediction by 1.47 (run 1 factor from ext sideband)
#
# Each step ADDS cuts on top of the previous one, so the cost/benefit of every
# handle is visible. Each combination of cut values gets its own folder, so
# trying ELCONF=8 then ELCONF=9 keeps both. A purity / efficiency / data-vs-pred table
# for all steps is printed at the end.
#   cutflow_run1/flash3.0_elconf8.0_prim4.0/<step>/...   (run 3: cutflow/...)
#
# The EXT weight and hygiene rules come from the tables themselves (each records
# its sample set), so nothing here needs to change between run 1 and run 3.
set -eu
AN=$(cd "$(dirname "$0")" && pwd)
K=$(cd "$AN/../../../.." && pwd)
TABLES=${TABLES:-tables_run1}
TAB=$AN/$TABLES
VARPLOTS=${VARPLOTS:-1}   # per-step variable plots (~20 s/step); 0 = skip

# numpy is only inside the container; use it directly if we're not already in it
if python3 -c "import numpy" 2>/dev/null; then RUN="python3"
else RUN="$K/run_in_tufts_pointcept_container.sh python3"; fi

CMN="--nue-npz $TAB/nue.npz --bnb-npz $TAB/bnb.npz \
     --ext-npz $TAB/ext.npz --data-npz $TAB/data.npz --ext-rescale ${EXTRESCALE:-1.0}"

# ---------------- cut values: edit these (or override from the env) --------
FLASH=${FLASH:-2.7}          # log10(flash chi2) upper bound
ELCONF=${ELCONF:-8.0}         # LArPID e-confidence       (signal is at HIGH values)
PRIM=${PRIM:-0.0}            # LArPID primariness        (kills Michel/delta e + EXT)
EL=${EL:--0.10}               # LArPID electron score log p_e (keep ABOVE; signal just below 0)
MU=${MU:--6.5}               # e-shower muon score       (keep BELOW)
EG=${EG:-4.0}                # egamma larpid score (keep ABOVE)
VTXMU=${VTXMU:--1.0}         # muon-like track sharing the e-vertex (keep BELOW)
NPHOT=${NPHOT:-0}            # max # BDT-passing photons at the vertex (pi0 veto)
EXTRESCALE=${EXTRESCALE:-1.0} # multiply the EXT weight (e.g. 1.3); 1.0 = spill-based
# one folder per combination of cut values, so trials never overwrite each other
TAGX=""; [ "$EXTRESCALE" != "1.0" ] && [ "$EXTRESCALE" != "1" ] && TAGX="_ext${EXTRESCALE}"
OUT=${OUT:-$AN/cutflow${TABLES#tables}/flash${FLASH}_elconf${ELCONF}_prim${PRIM}_el${EL}_mu${MU}_vtxmu${VTXMU}_nphot${NPHOT}${TAGX}}
# Other flags, all on top of the above (see nue_cc_common.add_cut_args):
#   --el-cut X  --egamma-cut X  --elconf-lf-cut X  --egamma-lf-cut X  --mu-lf-cut X
#   --vtxmu-lf-cut X  --nphoton-max N  --nnovtxphoton-max N
#   --objectness-cut X  --vtxscore-cut X  --vtxfraccosmic-cut X  --ext-scale W

LOG=$(mktemp); trap 'rm -f $LOG' EXIT
# Every step folder gets: reco_ele_energy, flashchi2, eff_vs_*, bg_truth_category
# (nue_cc_overlay.py) and, unless VARPLOTS=0, for EVERY candidate cut variable
# at that step (nue_cc_scan.py):
#   var_<v>.png   stacked prediction + data | efficiency & purity vs threshold
#   cat_<v>.png   the same variable stacked by what the e-candidate truly is
#   bg_composition.png, scan_summary.md (best next cut for each variable)
# -- i.e. what you read to decide which cut to try next.
step () {  # step <name> <cut flags...>
  local name=$1; shift
  echo "=================== ${name} ==================="
  $RUN $AN/nue_cc_overlay.py $CMN --plots $OUT/$name "$@" 2>&1 | tee -a $LOG.$name \
    | grep -E "SELECTION SUMMARY|purity|efficiency|data/pred|TOTAL" || true
  if [ "$VARPLOTS" = "1" ]; then
    $RUN $AN/nue_cc_scan.py $CMN --plots $OUT/$name --target-eff 0.4 "$@" >/dev/null 2>&1 \
      || echo "  !! variable plots failed for $name"
  fi
  echo "$name" >> $LOG
}

# ---------------- the cutflow: edit / comment / reorder these ---------------
#step 0_precut                               --flashchi2-cut -1
#step 1_vtxmu${VTXMU}                        --vtxmu-cut $VTXMU
#step 2_prim${PRIM}                          --vtxmu-cut $VTXMU --primariness-cut $PRIM
#step 3_el${EL}                              --vtxmu-cut $VTXMU --primariness-cut $PRIM --el-cut $EL
#step 4_mu${MU}                              --vtxmu-cut $VTXMU --primariness-cut $PRIM --el-cut $EL --mu-cut $MU
step 5_egamma${EG}                              --vtxmu-cut $VTXMU --primariness-cut $PRIM --el-cut $EL --mu-cut $MU --egamma-cut $EG
# step 1_flash${FLASH}                        --flashchi2-cut $FLASH
# step 2_elconf${ELCONF}                      --flashchi2-cut $FLASH --elconf-cut $ELCONF
# step 3_prim${PRIM}                          --flashchi2-cut $FLASH --elconf-cut $ELCONF --primariness-cut $PRIM
#step 4_mu${MU}                              --flashchi2-cut $FLASH --elconf-cut $ELCONF --primariness-cut $PRIM --mu-cut $MU
#step 4_vtxmu${VTXMU}                        --flashchi2-cut $FLASH --elconf-cut $ELCONF --primariness-cut $PRIM --vtxmu-cut $VTXMU
#step 4_nphot${NPHOT}                        --flashchi2-cut $FLASH --elconf-cut $ELCONF --primariness-cut $PRIM --ngoodphoton-max $NPHOT

# ---------------- summary table --------------------------------------------
echo
printf "%-28s %8s %8s %8s %8s %8s\n" step purity eff pred data data/pred
while read -r name; do
  f=$LOG.$name
  pur=$(grep -oP "purity \(nueCC/pred\)\s*=\s*\K[0-9.]+" $f || echo -)
  eff=$(grep -oP "efficiency \(sel/true-FV\)=?\s*=?\s*\K[0-9.]+" $f || echo -)
  pred=$(grep -oP "TOTAL pred\s+\K[0-9.]+" $f || echo -)
  dat=$(grep -oP "bnb5e19 data\s+\K[0-9]+" $f || echo -)
  dp=$(grep -oP "data/pred \K[0-9.]+" $f || echo -)
  printf "%-28s %8s %8s %8s %8s %8s\n" "$name" "$pur" "$eff" "$pred" "$dat" "$dp"
  rm -f $f
done < $LOG
echo "(LANTERN: purity 0.90 / eff 0.55)   plots -> $OUT"
