#!/bin/bash

AN=./
CMN="--nue-npz $AN/tables/nue.npz --bnb-npz $AN/tables/bnb.npz \
     --ext-npz $AN/tables/ext.npz --data-npz $AN/tables/data.npz"

FLASH_CUT_VAL="3.0"
VTXMU_CUT="-2.0"

# # step 0 — no cuts (disable flash cut with -1)
# python3 $AN/nue_cc_overlay.py $CMN \
#    --flashchi2-cut -1 --plots $AN/cutflow/0_precut

# # step 1 — flash cut
# python3 $AN/nue_cc_overlay.py $CMN \
#    --flashchi2-cut ${FLASH_CUT_VAL} --plots $AN/cutflow/1_flash

# # step 2 — + vertex-muon veto
# python3 $AN/nue_cc_overlay.py $CMN \
#    --flashchi2-cut ${FLASH_CUT_VAL} --vtxmu-lf-cut ${VTXMU_CUT} --plots $AN/cutflow/2_vtxmu

# # step 3 — + p_mu cut
# python3 $AN/nue_cc_overlay.py $CMN \
#    --flashchi2-cut ${FLASH_CUT_VAL} --vtxmu-lf-cut ${VTXMU_CUT} --mu-cut -3.0 --plots $AN/cutflow/3_probmucut

# step 4 — + primariness score
python3 $AN/nue_cc_overlay.py $CMN \
   --flashchi2-cut ${FLASH_CUT_VAL} --vtxmu-lf-cut ${VTXMU_CUT} --mu-cut -3.0 --primariness-cut 1.0 --plots $AN/cutflow/4_primaryness

# step 5 — + electron confidence
python3 $AN/nue_cc_overlay.py $CMN \
   --flashchi2-cut ${FLASH_CUT_VAL} --vtxmu-lf-cut -4.0 --mu-cut -3.0 --primariness-cut 1.0 --elconf-lf-cut 5.0 --mu-cut -6.0 --plots $AN/cutflow/5_elconf
