# Making a MicroBooNE Data Pipeline to test the Shower Origin Model

We want to test the effectiveness of the shower origin model in reconstructing showers
and aiding in identifying neutrino interactions with a signel photon. To do this we must
apply the shower origin model within a reconstruction pipeline for data coming
from the MicroBooNE detector. We describe in this document, the steps needed to do this,
and provide a detailed plan for implementation.

## Overview of Steps to integrate into LANTERN reco

  1. Make spacepoints from wire-plane image data, remove ghost points, and apply SSNet labels.
     This takes in ROOT data files and produces another ROOT data file.
     Both the input and output files will contain multiple events -- with the same number of events in
     each file. (Completed.)
  2. Convert the ROOT data file format into an h5 format that can be loaded by Pointcept/datasets/shower_origin/shower_origin.py.
     This will produce single h5 files with one event in each file. Each event will contain information on approximately 20-40
     shower fragents for which we need to make predictions.
  3. Apply the shower origin model to each of the event files. We list the paths to the input h5 event files in a textfile.
     The inference script will produce a single result h5 file for all of the events.
  4. Integrate the output into the lantern reco chain as an option module.
     The module reads in the output of step 1 and the output of step 3.
     The information from the shower fragment origin model is used to select neutrino-candidate fragments.
     We also use it to build the shower.

## Notes on Step 1

The input that will go into step one is made by taking (real or simulated) data from the experiment
and processing it through `ubdl/lantern_scripts.sh` which runs `ubdl/larflow/larmatchnet/larmatch/deploy_larmatchme.py`

Example of running this command on real data:

```
python3 deploy_larmatchme.py --config-file config_larmatchme_deploycpu.yaml --supera merged_dlana_d5cd7f5c-67e6-4bee-8c3a-dcefb42a63c0.root --weights /cluster/home/ubdl/larflow/larmatchnet/larmatch//larmatch_ckpt78k.pt --output output_test.root --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit -tb
```

Note that we run this in the production microboone "lantern" container that can be if one has access to MicroBooNE's CVMFS. 

The script `deploy_larmatchme.py` makes two ROOT files. They are the "larcv" and "larlite" ROOT files 
with names derived from the given `--output` command line argument but withthe terminal `.root` changes to 
`_larcv.root` and `_larlite.root`, respectively. 

The key outputs we need to make shower clusters for the shower origin model
are in the ROOT TTree `larflow3dhit_larmatch_tree`. The tree has a single branch which only one
element which is a container class called `larlite::data::event_larflow3dhit`. This class is 
essentially a wrapper of `std::vector<larlite::data::larflow3dhit>` and is a vector container of 
the class representing spacepoint information called `larflow::data::larflow3dhit`.
The header and source for this class can be found in `ubdl/larlite/larlite/DataFormat/larflow3dhit.h/.cxx`.
The hit itself inherits from `std::vector<float>` and stores a list of numbers to associate
with 3D spacepoints inside the MicroBooNE LArTPC.
The class is a bit abused as what is stored in the float is dependent on the class that makes it.

The class that makes the `larflow::data::larflow3dhit` instances we will use is 
the class `larflow::prep::FlowMatchHitMaker`.
Its source is in `ubdl/larflow/larflow/PrepFlowMatchData/FlowMatchHitMaker.h/.cxx`. Specifically,
the class method that makes and stores hits is `FlowMatchHitMaker::make_hits`.
From the documentation in that class the vector of floats stored for each `larflow3dhit` is

```
   * larflow3dhit inherits from vector<float>. The values in the vector are as follows:
   * [0-2]:   x,y,z
   * [3-9]:   7 flow direction scores + 1 max score (deprecated based on 2-flow paradigm. for triplet, [9] is the only score stored 
   * [10-16]: 7 ssnet scores, (bg,track,shower), from larmatch (not 2D sparse ssnet)
   * [17-22]: 6 keypoint label score [nu,track-start,track-end,nu-shower,delta,michel]
   * [23-25]: reserved for plane charge
   * [26-28]: 3D flow direction
```

It is this data that we will want to convert into the hdf5 format compatible with the shower origin dataset class.

In order to gain access to the information in the ROOT file containing the `larflow3dhit` objects, it is best to use
the larlite IO manager. Here is an example of the python commands needed to get the data for an event:

```
import ROOT as rt
from larlite import larlite

inputfile = "output_larlite.root"

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( inputfile )
io.open() # initialize the IO manager

nentries = io.get_entries()

for ientry in range(nentries):
  event_hits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
  nhits = event_hits.size()
  for ihit in range(nhits):
    hit = event_hits.at(ihit)
    ...

```

The header and source code for the `larlite::storage_manager` class is in `ubdl/larlite/larlite/DataFormat/storage_manager.h/.cxx`


## Analyzing the output before full integration

Before performing Step 4, we can make data up to step 3. We then make a tree to merge with lantern ana.
We use it to help "select" an event. We ask if an inside shower event exists.
We also ask if the predicted origin satisfies certain requirements. (We define the selection criterion later.)