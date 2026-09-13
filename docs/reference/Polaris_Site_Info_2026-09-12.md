# Polaris site info gathered 2026-09-12 (terminal transcript, user-run commands)

```
twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> ls /eagle/neutrinoGPU/twongj01/ /eagle/neutrinoGPU/twongj01/polaris_assets /eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19
/eagle/neutrinoGPU/twongj01/:
data  larformer_pointcept  pointcept_cuml.sif  polaris_assets  prongCNN

/eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19:
squashfs

/eagle/neutrinoGPU/twongj01/polaris_assets:
env.sh  kpv2_assets  MANIFEST.txt  oldrepo_assets  pointcept_cuml.sif  sha256sums.txt


---

env.sh  kpv2_assets  MANIFEST.txt  oldrepo_assets  pointcept_cuml.sif  sha256sums.txt
twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> head -8 /eagle/neutrinoGPU/twongj01/polaris_assets/env.sh
# source this on the far side after cloning the repo and copying the assets.
# 1) the repo clone (must contain the assets' kpv2_assets/* files at the same
#    relative paths, or point LARFORMER_KPV2_ROOT at the kpv2_assets dir itself
#    -- the configs only read sonata/, exp/ and lartpc/flashmatch/data/ from it)
export LARFORMER_KPV2_ROOT=${LARFORMER_KPV2_ROOT:-/eagle/neutrinoGPU/twongj01/uboone/assets/kpv2_assets}
# 2) the old-repo assets (sonata pretrain loaded into the backbones at build time)
export LARFORMER_OLD_REPO=${LARFORMER_OLD_REPO:-/eagle/neutrinoGPU/twongj01/uboone/assets/oldrepo_assets}
export LARFORMER_BATTERY_SLICER_CKPT=$LARFORMER_KPV2_ROOT/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth

---

36aed2a (HEAD -> nutufts_lartpc_keypointdev_v2, origin/nutufts_lartpc_keypointdev_v2) larpid: PRONGCNN_DIR env override; polaris handoff notes the prongCNN dependency

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> module avail apptainer 2>&1 | tail -5; ml use /soft/modulefiles; ml spack-pe-base; ml apptainer; apptainer --version

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


apptainer version 1.4.1

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> qstat -Qf debug prod small preemptable 2>/dev/null | grep -E "^Queue|resources_(max|min)\.(nodect|walltime)|max_run|max_queued"
Queue: debug
    resources_max.nodect = 2
    resources_max.walltime = 01:00:00
    resources_min.nodect = 1
    resources_min.walltime = 00:05:00
    max_run = [u:PBS_GENERIC=1]
    max_run_res.nodect = [o:PBS_ALL=24]
Queue: prod
    max_queued = [p:PBS_GENERIC=100]
    resources_max.nodect = 496
    resources_max.walltime = 24:00:00
    resources_min.nodect = 10
    resources_min.walltime = 00:05:00
Queue: small
    max_queued = [p:PBS_GENERIC=10]
    resources_max.nodect = 24
    resources_max.walltime = 03:00:00
    resources_min.nodect = 10
    resources_min.walltime = 00:05:00
Queue: preemptable
    max_queued = [p:PBS_GENERIC=20]
    max_queued = [u:PBS_GENERIC=20]
    resources_max.nodect = 10
    resources_max.walltime = 72:00:00
    resources_min.nodect = 1
    max_run = [p:PBS_GENERIC=10]
    max_run = [u:cseitz=10]

---

did not have any past jobs

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> sbank-list-allocations -p neutrinoGPU 2>/dev/null || sbank allocations list -p neutrinoGPU
 Allocation  Suballocation  Start       End         Resource  Project      Jobs  Charged  Available Balance 
 ----------  -------------  ----------  ----------  --------  -----------  ----  -------  ----------------- 
 14269       14177          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.0 
 14269       14453          2025-07-01  2026-10-01  polaris   neutrinoGPU     1     0.05              231.4 
 14269       14608          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0               0.09 
 14269       14609          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0               0.08 
 14269       15579          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.1 
 14269       15633          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -281.1 
 14269       15858          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              147.0 
 14269       16030          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.0 
 14269       16343          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              239.5 
 14269       16540          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              127.3 
 14269       16742          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              100.0 
 14269       16743          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -110.5 
 14269       16996          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -296.3 
 14269       17114          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0           -2,800.5 
 14269       17253          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              -74.4 
 17248       17256          2026-08-27  2026-10-01  polaris   neutrinoGPU     0      0.0            2,323.1 

Totals:
  Rows: 16
  Polaris:
    Available Balance: -394.1 Node Hours
    Charged          : 0.05 Node Hours
    Jobs             : 1 

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> lfs getstripe -d /eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19 | head -3
stripe_count:  1 stripe_size:   1048576 pattern:       0 stripe_offset: -1

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> lfs getstripe -d /eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs/ | 
head -3
stripe_count:  1 stripe_size:   1048576 pattern:       0 stripe_offset: -1

--

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> ls /eagle/neutrinoGPU/twongj01/prongCNN/checkpoints /eagle/neutrinoGPU/twongj01/prongCNN/models
/eagle/neutrinoGPU/twongj01/prongCNN/checkpoints:
LArPID_alternate_network_weights.pt  LArPID_default_network_weights.pt

/eagle/neutrinoGPU/twongj01/prongCNN/models:
datasets.py                                          datasets_reco_5ClassSoftLabel.py             models_instanceNorm_reco_2chan_tripleTask_forSHAP.py
datasets_reco_5ClassHardLabel_multiTask.py           datasets_reco.py                             models_instanceNorm_reco_2chan_tripleTask.py
datasets_reco_5ClassHardLabel.py                     models_instanceNorm.py                       models.py
datasets_reco_5ClassHardLabel_quadTask_multifile.py  models_instanceNorm_reco_1chan.py            models_setBNMom.py
datasets_reco_5ClassHardLabel_quadTask.py            models_instanceNorm_reco_2chan_multiTask.py  normalization_constants.py
datasets_reco_5ClassHardLabel_tripleTask_forSHAP.py  models_instanceNorm_reco_2chan.py
datasets_reco_5ClassHardLabel_tripleTask.py          models_instanceNorm_reco_2chan_quadTask.py

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> sbank-list-allocations -r polaris -p neutrinoGPU -f "+subname users_list"
 Allocation  Suballocation  Start       End         Resource  Project      Jobs  Charged  Available Balance  Subname                  Users                                              
 ----------  -------------  ----------  ----------  --------  -----------  ----  -------  -----------------  -----------------------  -------------------------------------------------- 
 14269       14177          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.0  prod                     None                                               
 14269       14453          2025-07-01  2026-10-01  polaris   neutrinoGPU     1     0.05              231.4  debug                    twester, nathanielerowe, abhat, rlazur, nupur      
 14269       14608          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0               0.09  ICARUSRun4_Summer2025    twester                                            
 14269       14609          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0               0.08  SPINE_Summer2025         twester, rlazur                                    
 14269       15579          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.1  SPINE_Winter2025         twester                                            
 14269       15633          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -281.1  SPINE_Train2026          nathanielerowe, jmueller, nupur, dihans            
 14269       15858          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              147.0  PROFit_2026              nathanielerowe                                     
 14269       16030          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0                0.0  SBND_BSM_Spring2026      twester                                            
 14269       16343          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              239.5  SPINE_Detsys_Spring2026  twester, jmueller, nupur, msiden                   
 14269       16540          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              127.3  SPINE_TrainExotics2026   jmueller, aliciavr                                 
 14269       16742          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              100.0  DetsimGPU                oalterka                                           
 14269       16743          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -110.5  ICARUSRun4_Summer2026    jmueller, nupur, dcarber, aliciavr, dihans, msiden 
 14269       16996          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0             -296.3  ICARUSRun2_Summer2026    jmueller, nupur, dcarber, aliciavr, dihans, msiden 
 14269       17114          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0           -2,800.5  SpineOsc_Summer2026      jmueller, nupur, dcarber, aliciavr, dihans, msiden 
 14269       17253          2025-07-01  2026-10-01  polaris   neutrinoGPU     0      0.0              -74.4  SpineOsc_Summer2026_2    jmueller, nupur, dcarber, aliciavr, dihans, msiden 
 17248       17256          2026-08-27  2026-10-01  polaris   neutrinoGPU     0      0.0            2,323.1  None                     None                                               

Totals:
  Rows: 16
  Polaris:
    Available Balance: -394.1 Node Hours
    Charged          : 0.05 Node Hours
    Jobs             : 1 


---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> ls /eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs/ | head; ls /eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs/*.sqfs | wc -l
bnb5e19_merged_sp_000.sqfs
bnb5e19_merged_sp_000.sqfs.manifest
bnb5e19_merged_sp_000.sqfs.ok
bnb5e19_merged_sp_001.sqfs
bnb5e19_merged_sp_001.sqfs.manifest
bnb5e19_merged_sp_001.sqfs.ok
bnb5e19_merged_sp_002.sqfs
bnb5e19_merged_sp_002.sqfs.manifest
bnb5e19_merged_sp_002.sqfs.ok
bnb5e19_merged_sp_003.sqfs
12

---

twongj01@polaris-login-02:/eagle/neutrinoGPU/twongj01/larformer_pointcept> (cd /eagle/neutrinoGPU/twongj01/polaris_assets && head -3 sha256sums.txt) 
ef22df9d8b3aabc59d1fe3bfd31542dadf40e24f48c76398fab02ded65485b63  kpv2_assets/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth
0452af01fb1d7d21b98826ce32d7760ff26738828ceb2fb6e8149bf3a6689921  kpv2_assets/exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth
bd800de10ded4b6ac9e2a7a031cf5a6ca17ecf84e322689083755d1f12855b6a  kpv2_assets/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth
```
