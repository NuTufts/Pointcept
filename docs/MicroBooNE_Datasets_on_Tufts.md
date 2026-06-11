# MicroBooNE Datasets on Tufts

Data sets on the Tufts cluster are used to study the effectiveness of the models to analyze MicroBooNE data.

We split them into different named datasets, corresponding to different detector conditions and simulation configurations.
Each dataset has a textfile with a list of files belong to that dataset.

## MicroBooNE Production Datasets 

These datasets come from the MicroBooNE simulation and real data production workflow. The files are in the ROOT format. The data is stored in the form of c++ classes whose instances are serialized. 

- mcc9_v29e_dl_run3b_bnb_nu_overlay
  - Simulated neutrino events assuming the full BNB neutrino flux. Nu interaction mixed with real detector data from events recorded when the beam is off.
  - Neutrino interactions are generated in the liquid argon volume inside the cryostat. Includes events outside the active TPC volume.
  - Run 3 detector state
  - input list: /cluster/tufts/wongjiradlabnu/mrosen25/filelists/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge_filelist.txt

- mcc9_v29e_dl_run3_G1_extbnb_dlreco
  - Real data when the beam is off
  - Run 3 detector state
  - input list: /cluster/tufts/wongjiradlabnu/mrosen25/filelists/mcc9_v29e_dl_run3_G1_extbnb_dlreco_processed_filelist.txt

- mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay
  - Simulated neutrino events assuming BNB neutrino flux and including only charged-current electron neutrino interactions. 
  - Simulated interaction mixed with real detector data from events recorded when the beam is off.
  - Neutrino interactions are generated in the liquid argon volume inside the TPC.
  - input list: /cluster/tufts/wongjiradlabnu/mrosen25/filelists/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge_filelist.txt

### Accessing the data from these files

Because the data is stored as C++ classes, we need to load certain libraries to be able to access and read the data.

On the Tufts cluster, we load the 'ubdl' respository.

Inside the pointcept container, see `run_in_tufts_pointcept_container.sh`, setup the bash environment by:

```
# Setup the repository
source /cluster/tufts/wongjiradlabnu/larbys/gen2/pointcept_env/ubdl/setenv_pointcept_container.sh
```

To access the image data in a python script:

```
from larcv import larcv

# Official MicroBooNE data store the images in the "tick-backward" format where
# the waveform on one wire is stored from last tick to first tick.
tick_dir = larcv.IOManager.kTickBackward

iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv", tick_dir)
iolcv.add_in_file("filepath1")
iolcv.add_in_file("filepath2")

for prod in ("wire", "wiremc", "thrumu", "ancestor", "segment","instance", "larflow"):
    iolcv.specify_data_read("image2d", prod)
iolcv.specify_data_read("chstatus", "wire")
iolcv.specify_data_read("chstatus", "wiremc")
# we flip the tick order of image products to be tick-forward
print("REVERSE TICK-ORDER OF LARCV DATA")
iolcv.reverse_all_products()
iolcv.initialize()

for entry in range(iolcv.get_n_entries()):
    iolcv.read_entry(entry)
    # get wire plane image data
    event_wireplane_images = io.get_data(larcv.kProductImage2D, "wire")
    # induction plane U
    uplane = event_wireplane_images.as_vector()[0]
    # induction plane V
    vplane = event_wireplane_images.as_vector()[1]
    # collection plane Y
    yplane = event_wireplane_images.as_vector()[2]
    ...

# Close the IOManager
iolcv.close()
```

A copy of the `larcv` library code on the Tufts cluster is at `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/larcv/larcv/core/`. The c++ class definitions are in `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/larcv/larcv/core/DataFormat/`.

To access meta-data, such as a list of the particles created in the simulation:

```
from larlite import larlite

ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
ioll.add_in_file("filepath1")
ioll.add_in_file("filepath2")
ioll.open()

for ientry in range(ioll.get_n_entries()):
    ioll.go_to(ientry)
    ...

ioll.close()


```

A copy of the `larlite` library code on the Tufts cluster is at `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/larlite/larlite/`. The c++ class definitions are in `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/larlite/larlite/DataFormat/`.

These files contain both larcv and larlite data (hence why they are often referred to as "merged" files).

Each data product is stored in single-branch ROOT TTree objects. For example, the "wire" data product is stored in a TTree called "wire" and the "wiremc" data product is stored in a TTree called "wiremc". 

Below is an example of the TTree names in one of the merged files. We've truncated the whole list to the most relevant trees.
Many of the trees contain data from an old reconstruction workflow that is now deprecated.

```
twongj01@login-p01:/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env$ root /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/41/21/merged_dlreco_06ae2262-83dc-4375-a9a6-e74e63f55849.root
   ------------------------------------------------------------------
  | Welcome to ROOT 6.36.06                        https://root.cern |
  | (c) 1995-2025, The ROOT Team; conception: R. Brun, F. Rademakers |
  | Built for linuxx8664gcc on Dec 02 2025, 14:30:27                 |
  | From tags/v6-36-06@v6-36-06                                      |
  | With c++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0                   |
  | Try '.help'/'.?', '.demo', '.license', '.credits', '.quit'/'.q'  |
   ------------------------------------------------------------------

root [0] 
Attaching file /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/41/21/merged_dlreco_06ae2262-83dc-4375-a9a6-e74e63f55849.root as _file0...
(TFile *) 0x55a6915610b0
root [1] .ls
TFile**         /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/41/21/merged_dlreco_06ae2262-83dc-4375-a9a6-e74e63f55849.root
 TFile*         /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/41/21/merged_dlreco_06ae2262-83dc-4375-a9a6-e74e63f55849.root
  KEY: TTree    image2d_wire_tree;1     wire tree
  KEY: TTree    chstatus_wire_tree;1    wire tree
  KEY: TTree    image2d_ubspurn_plane0_tree;1   ubspurn_plane0 tree
  KEY: TTree    image2d_ubspurn_plane1_tree;1   ubspurn_plane1 tree
  KEY: TTree    image2d_ubspurn_plane2_tree;1   ubspurn_plane2 tree
  KEY: TTree    image2d_thrumu_tree;1   thrumu tree
  KEY: TTree    image2d_segment_tree;1  segment tree
  KEY: TTree    image2d_instance_tree;1 instance tree
  KEY: TTree    image2d_ancestor_tree;1 ancestor tree
  KEY: TTree    partroi_segment_tree;1  segment tree
  KEY: TTree    image2d_larflow_tree;1  larflow tree
  ...
  KEY: TTree    larlite_id_tree;1       LArLite Event ID Tree
  KEY: TTree    gtruth_generator_tree;1 gtruth Tree by generator
  KEY: TTree    mctruth_corsika_tree;1  mctruth Tree by corsika
  KEY: TTree    mctruth_generator_tree;1        mctruth Tree by generator
  KEY: TTree    mcflux_generator_tree;1 mcflux Tree by generator
  KEY: TTree    mcshower_mcreco_tree;1  mcshower Tree by mcreco
  KEY: TTree    daqheadertimeuboone_daq_tree;1  daqheadertimeuboone Tree by daq
  KEY: TTree    hit_gaushit_tree;1      hit Tree by gaushit
  KEY: TTree    hit_portedThresholdhit_tree;1   hit Tree by portedThresholdhit
  KEY: TTree    crthit_crthitcorr_tree;1        crthit Tree by crthitcorr
  KEY: TTree    crttrack_crttrack_tree;1        crttrack Tree by crttrack
  KEY: TTree    ophit_ophitBeam::OverlayStage1OpticalDLrerun_tree;1     ophit Tree by ophitBeam::OverlayStage1OpticalDLrerun
  KEY: TTree    ophit_ophitBeamCalib_tree;1     ophit Tree by ophitBeamCalib
  KEY: TTree    ophit_ophitCosmic::OverlayStage1OpticalDLrerun_tree;1   ophit Tree by ophitCosmic::OverlayStage1OpticalDLrerun
  KEY: TTree    ophit_ophitCosmicCalib_tree;1   ophit Tree by ophitCosmicCalib
  KEY: TTree    opflash_opflashBeam_tree;1      opflash Tree by opflashBeam
  KEY: TTree    opflash_opflashCosmic_tree;1    opflash Tree by opflashCosmic
  KEY: TTree    opflash_simpleFlashBeam_tree;1  opflash Tree by simpleFlashBeam
  KEY: TTree    opflash_simpleFlashBeam::OverlayStage1OpticalDLrerun_tree;1     opflash Tree by simpleFlashBeam::OverlayStage1OpticalDLrerun
  KEY: TTree    opflash_simpleFlashCosmic_tree;1        opflash Tree by simpleFlashCosmic
  KEY: TTree    opflash_simpleFlashCosmic::OverlayStage1OpticalDLrerun_tree;1   opflash Tree by simpleFlashCosmic::OverlayStage1OpticalDLrerun
  KEY: TTree    sps_portedSpacePointsThreshold_tree;1   sps Tree by portedSpacePointsThreshold
  ...
  KEY: TTree    trigger_triggersim_tree;1       trigger Tree by triggersim
  KEY: TTree    mctrack_mcreco_tree;1   mctrack Tree by mcreco
  KEY: TTree    ass_inter_ass_tree;1    ass Tree by inter_ass
  KEY: TTree    ass_opflashBeam_tree;1  ass Tree by opflashBeam
  KEY: TTree    ass_opflashCosmic_tree;1        ass Tree by opflashCosmic
  KEY: TTree    ass_portedSpacePointsThreshold_tree;1   ass Tree by portedSpacePointsThreshold
  KEY: TTree    ass_simpleFlashBeam_tree;1      ass Tree by simpleFlashBeam
  KEY: TTree    ass_simpleFlashBeam::OverlayStage1OpticalDLrerun_tree;1 ass Tree by simpleFlashBeam::OverlayStage1OpticalDLrerun
  KEY: TTree    ass_simpleFlashCosmic_tree;1    ass Tree by simpleFlashCosmic
  KEY: TTree    ass_simpleFlashCosmic::OverlayStage1OpticalDLrerun_tree;1       ass Tree by simpleFlashCosmic::OverlayStage1OpticalDLrerun
  ...
  KEY: TTree    mceventweight_eventweight4to4aFix_tree;1        mceventweight Tree by eventweight4to4aFix
  KEY: TTree    mceventweight_eventweightLEE_tree;1     mceventweight Tree by eventweightLEE
  KEY: TTree    swtrigger_swtrigger_tree;1      swtrigger Tree by swtrigger
  ...
  KEY: TTree    potsummary_generator_tree;1     potsummary Tree by generator
  ...
  KEY: TTree    sparseimg_sparseuresnetout_tree;1       sparseuresnetout tree
```

### Helper Classes for parsing the simulation true

We have developed classes to help parse the simulation true information in the MicroBooNE production datasets. 

See the `SimChTripletLabelMaker` class in `ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.h`.

TODO: Summarize what this class does. List the TTrees it loads and the classes it uses to organize the data.

If interested in the simulation meta data, i.e. the particles in the simulation, and not how they relate to the image or spacepoint data, one can use the class `MCParticleGraph` (header in `ubdl/ublarcvapp/ublarcvapp/MCTools/MCParticleGraph.h`) to collect the various truth information in the larlite data TTrees.

TODO: Provide more information on how to load the truth information using `MCParticleGraph` here.

## Flattened Ntuple Datasets

The data in the MicroBooNE production datasets can be difficult to handle because of their use of custom C++ classes.
Therefore, we also have flattened ntuple versions of the datasets. 

These files are located at `/cluster/tufts/wongjiradlabnu/nutufts/data/ntuples/`.

List of files for the datasets:

- mcc9_v29e_dl_run3b_bnb_nu_overlay: `dlgen2_reco_v2me05_gen2ntuple_v7_run3b_bnb_nu_overlay_nocrtremerge.root`
- mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay: `dlgen2_reco_v2me05_ntuple_v7_mcc9_run3b_bnb_intrinsic_nue_overlay_nocrtremerge.root`
- mcc9_v29e_dl_run3_G1_extbnb_dlreco: `dlgen2_reco_v2me06_ntuple_v5_mcc9_v29e_dl_run3_G1_extbnb_dlreco.root`

Each file has two TTree objects:
- EventTree: Contains the event-level information.
- potTree: For simulation datasets, this tree contains the simulated livetime of the dataset in terms of protons on target (POT). It has the POT for each file in the simulated dataset. To get the total POT, the `totGoodPOT` branch should be summed across all entries.

The list of branches in the ntuple and their description can be found in the [gen2ntuple repository README](https://github.com/NuTufts/gen2ntuple/blob/main/README.md).









