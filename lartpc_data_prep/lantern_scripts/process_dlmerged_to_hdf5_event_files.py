import os,sys
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Process larcv/larlite into HDF5 entry data files.')
parser.add_argument("-i","--input-dlmerged",required=True,type=str,help="Input dlmerged file")
parser.add_argument("-v","--verbosity",type=int,default=2,help="Verbosity level from normal=2 to debug=0")
parser.add_argument("-n","--nentries",type=int,default=-1,help="Number of entries to run. (default is -1, which will run all entries in the file.)")
parser.add_argument('-d','--is-data',default=False,action='store_true',help='if provided, set to run in data mode (no simulation information processed.)')
parser.add_argument('-tb','--tick-backward',default=False,action='store_true',help='if flag provided, assume larcv data is stored in tick-backward format and reverse upon loading.')
parser.add_argument('--mcc9',default=False,action='store_true',help='if provided, set to run in mcc9 mode (use mcc9 truth information).')
parser.add_argument('--adc',default="wiremc",type=str,help="ADC image name to use for the input images.")
parser.add_argument('--fileno-tag',default="",type=str,help="Optional tag (e.g. 'fileno00001') inserted into output filenames between the 'pointceptdata_' prefix and the input basename, to disambiguate intermediates produced from different input files.")

args = parser.parse_args(sys.argv[1:])

import ROOT
from larlite import larlite
from larcv import larcv
from larflow import larflow

dlmerged_input = args.input_dlmerged

start_entry = 0
end_entry = args.nentries

# Setup algorithm
simchmaker = larflow.prep.SimChTripletLabelMaker()
simchmaker.set_verbosity(args.verbosity)
simchmaker.save_truth_tripletinfo( True )
simchmaker.set_adc_treename( args.adc )
if args.is_data:
  print("Running in DATA mode")
  simchmaker.set_is_data()

# setup truth point label maker
simchmaker._mcpixelmaker.set_verbosity(args.verbosity)
# configure to use wirecell driftWC simch
simchmaker._mcpixelmaker.set_driftwc_source()
# how much do we bleed out the truth labels?
simchmaker._mcpixelmaker.set_dwire(1)
simchmaker._mcpixelmaker.set_drow(0)
if args.mcc9:
  print("RUNNING IN MCC9 MODE")
  simchmaker.process_mcc9_sim()

# set keypoint maker verbosity
#simchmaker._mckpmaker.set_verbosity(args.verbosity)
simchmaker._shower_fragment_maker.set_verbosity(args.verbosity)

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( dlmerged_input )
ioll.set_verbosity(2)
ioll.open()

tick_direction=larcv.IOManager.kTickForward
if args.tick_backward:
  tick_direction=larcv.IOManager.kTickBackward 

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", tick_direction )
iolcv.add_in_file( dlmerged_input )
iolcv.specify_data_read( "image2d", "wire" )
iolcv.specify_data_read( "image2d", "wiremc" )
iolcv.specify_data_read( "image2d", "thrumu" )
iolcv.specify_data_read( "image2d", "ancestor" )
iolcv.specify_data_read( "image2d", "segment" )
iolcv.specify_data_read( "image2d", "instance" )
iolcv.specify_data_read( "image2d", "larflow" )
iolcv.specify_data_read( "chstatus", "wire" )
iolcv.specify_data_read( "chstatus", "wiremc" )
iolcv.specify_data_read( "image2d", "ubspurn_plane0" )
iolcv.specify_data_read( "image2d", "ubspurn_plane1" )
iolcv.specify_data_read( "image2d", "ubspurn_plane2" )
iolcv.specify_data_read( "sparseimage", "sparseuresnetout" )
iolcv.specify_data_read( "sparseimage", "sparsessnet" ) 

if args.tick_backward:
  print("REVERSE TICK-ORDER OF LARCV DATA")
  iolcv.reverse_all_products()

iolcv.set_verbosity(0)
iolcv.initialize()

nentries = ioll.get_entries()
if end_entry<0 or end_entry>=nentries:
  end_entry = nentries
else:
  end_entry = end_entry

# process input file name. we will use it to name the invididual event files
basefilename = os.path.basename( args.input_dlmerged )
# remove .root extension if its there
if ".root" in basefilename and basefilename[-5:]==".root":
  basefilename = basefilename[:-5]

for ientry in range(start_entry,end_entry):

  ioll.go_to(ientry)
  iolcv.read_entry(ientry)

  # we save one entry per file for use in Pointcept
  tag_part = f"{args.fileno_tag}_" if args.fileno_tag else ""
  entryfile_name = f"pointceptdata_{tag_part}{basefilename}_entry{ientry:06d}.h5"
  print(f"[{ientry}] output filename: ",entryfile_name)
  
  simchmaker.process( ioll, iolcv )
  #simchmaker._mckpmaker.printKeypoints()

  # save entry
  hdf_entry_prefix = f"/entry_0"  
  simchmaker.open_hdf_file( entryfile_name )
  simchmaker.save_entry( hdf_entry_prefix )
  simchmaker.close_hdf_file()

simchmaker.close_hdf_file()
