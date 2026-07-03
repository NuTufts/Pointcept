#!/bin/bash
#
# submit_score_3k.sh — score the 5 deghost/rescue sweep arms (eff + purity on the
# mixed 3000-event sample) on the BATCH partition, one task per arm, so it's
# robust to a dropped interactive session. Each task runs analyze_1g0X.py and
# writes:
#   - workdir_scale/cmpfix_<arm>.csv      (per-event reco labels)
#   - workdir_scale/scorefix_<arm>.txt    (one-line parseable: arm eff pur TP FP FN reco)
#   sbatch submit_score_3k.sh

#SBATCH --job-name=sp_scorefix
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/scorefix/score.%A_%a.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/scorefix/score.%A_%a.err

SP=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/single_photon
POINTCEPT=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
ENV=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v29e_dl_run3b_bnb_nu_overlay
OUTBASE=${DATADIR}/sp_compare_3k_fixed
TRUTH=workdir_scale/visible_photon_events.csv

ARMS=(base dg0p4 dg0p3 rescue dg0p3_resc base_dup)
a=${ARMS[$SLURM_ARRAY_TASK_ID]}
mkdir -p ${SP}/slurm/logs/scorefix
echo "scoring arm ${a}  (task ${SLURM_ARRAY_TASK_ID})"

module load apptainer/1.4.0 2>/dev/null || true
apptainer exec --bind /cluster:/cluster ${SIF} bash -c \
  "source ${ENV} >/dev/null 2>&1; export PYTHONPATH=${POINTCEPT}:${SP}:\$PYTHONPATH; cd ${SP}; \
   python3 analyze_1g0X.py --pred-dir ${OUTBASE}/${a} --truth-csv ${TRUTH} \
       --out workdir_scale/cmpfix_${a}.csv 2>&1 | tee /tmp/score_${a}.out; \
   python3 - <<PY
import re
t=open('/tmp/score_${a}.out').read()
def g(p,d='?'):
    m=re.search(p,t); return m.group(1) if m else d
eff=g(r'EFFICIENCY.*: ([0-9.]+)'); pur=g(r'PURITY.*: ([0-9.]+)')
reco=g(r'reco  1g0X\s+: (\d+)'); m=re.search(r'TP / FP / FN / TN : (\d+) / (\d+) / (\d+) / (\d+)',t)
TP,FP,FN=(m.group(1),m.group(2),m.group(3)) if m else ('?','?','?')
open('workdir_scale/scorefix_${a}.txt','w').write(
    '%-12s eff=%s pur=%s TP=%s FP=%s FN=%s reco=%s\n'%('${a}',eff,pur,TP,FP,FN,reco))
print('wrote scorefix_${a}.txt:', open('workdir_scale/scorefix_${a}.txt').read().strip())
PY"
echo "arm ${a} DONE"
