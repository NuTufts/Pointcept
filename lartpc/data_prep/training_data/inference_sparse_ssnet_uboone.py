from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys,argparse
from ctypes import c_int
import numpy as np
import torch
import traceback

# sparse uresnet imports (in networks/sparse_ssnet)
import uresnet
from uresnet.flags      import URESNET_FLAGS
from uresnet.main_funcs import inference
from uresnet.trainval   import trainval

from scipy.signal import convolve2d

"""
A convenience wrapper around SLAC's sparse uresnet. Used for deploying on SBN detectors.
"""

def forwardpass( plane, nrows, ncols, sparse_bson_list, weights_filepath ):

    from ROOT import std
    from larcv import larcv
    larcv.json.load_jsonutils()
    
    print("[SparseSSNet] forwardpass")
    # Get Configs going:
    # configuration from Ran:
    """
    inference --full -pl 1 -mp PATH_TO_Plane1Weights-13999.ckpt -io larcv_sparse 
    -bs 64 -nc 5 -rs 1 -ss 512 -dd 2 -uns 5 -dkeys wire,label 
    -mn uresnet_sparse -it 10 -ld log/ -if PATH_TO_INPUT_ROOT_FILE
    """
    print("[SparseSSNet] create uresnet_flags object")
    try:
        sys.argv = ["inference_sparse_ssnet.py"]
        config = uresnet.flags.URESNET_FLAGS()
    except Exception as e:
        print("[SparseSSNet] error creating URESNET_FLAGS: {}".format(sys.exc_info()[0]))
        print("[SparseSSNet] trackback")
        print(traceback.format_exc())
        return None
    print("[SparseSSNet] set options dictionary")
    config_args = { "full":True,               # --full
                    "plane":plane,        # -pl
                    "model_path":weights_filepath,  # -mp
                    "io_type":"larcv_sparse",  # -io
                    "batch_size":1,            # -bs
                    "num_class":5,             # -nc
                    "uresnet_filters":32,      # -uf
                    "report_step":1,           # -rs
                    "spatial_size":512,        # -ss
                    "data_dim":2,              # -dd
                    "uresnet_num_strides": 5,  # -uns
                    "data_keys":"wire,label",  # -dkeys
                    "model_name":"uresnet_sparse", # -mn
                    "iteration":1,            # -it
                    "log_dir":"./log/",          # -ld
                    "input_file":"none" }      # -if

    print("[SparseSSNet] update/set config")
    config.update(config_args)
    config.SPATIAL_SIZE = (nrows,ncols)
    config.TRAIN = False
        
    print("\n\n-- CONFIG --")
    for name in vars(config):
        attribute = getattr(config,name)
        if type(attribute) == type(config.parser): continue
        print("%s = %r" % (name, getattr(config, name)))

    # Set random seed for reproducibility
    #np.random.seed(config.SEED)
    #torch.manual_seed(config.SEED)

    print("[SparseSSNet] create trainval interface")
    interface = trainval(config)
    print("[SparseSSNet] initialize")
    interface.initialize()
    print("Loaded sparse pytorch_uresnet plane={}".format(plane))

    # parse the input data, loop over pybyte objects
    sparsedata_v = []
    rseid_v = []
    npts_v  = []
    ntotalpts = 0

    try:
        for bson in sparse_bson_list:
            c_run    = c_int()
            c_subrun = c_int()
            c_event  = c_int()
            c_id     = c_int()

            imgdata = larcv.json.sparseimg_from_bson_pybytes(bson,
                                                             c_run, 
                                                             c_subrun, 
                                                             c_event, 
                                                             c_id )
            npts = int(imgdata.pixellist().size()/(imgdata.nfeatures()+2))
            ntotalpts += npts
            sparsedata_v.append(imgdata)
            npts_v.append( npts )
            rseid_v.append( (c_run.value, c_subrun.value, c_event.value, c_id.value) )
        
        # make batch array
        batch_np = np.zeros( ( ntotalpts, 4 ) )
        startidx = 0
        idx      = 0
        for npts,img2d in zip( npts_v, sparsedata_v ):
            endidx   = startidx+npts
            #spimg_np = larcv.as_sparseimg_ndarray( img2d, larcv.msg.kNORMAL )
            spimg_np = larcv.as_sparseimg_ndarray( img2d, 2 )
            #print("spimg_np: {}".format(spimg_np[:,0:2]))

            # coords
            batch_np[startidx:endidx,0] = nrows-1-spimg_np[:,0] # tick
            batch_np[startidx:endidx,1] = spimg_np[:,1] # wire

            batch_np[startidx:endidx,2] = idx           # batch index
            batch_np[startidx:endidx,3] = spimg_np[:,2] # pixel value
            #print("batch_np: {}".format(batch_np[:,0:2]))
            idx += 1

        # pass to network
        data_blob = { 'data': [[batch_np]] }
        results = interface.forward( data_blob )
    except:
        print("[SparseSSNet] error converting msg/running net: {}".format(sys.exc_info()[0]))
        print("[SparseSSNet] trackback")
        print(traceback.format_exc())
        return None


    bson_reply = []
    startidx = 0
    try:
        for idx in range(len(results['softmax'])):
            ssnetout_np = results['softmax'][idx]
            #print("ssneout_np: {}".format(ssnetout_np.shape))
            rseid = rseid_v[idx]
            meta  = sparsedata_v[idx].meta(0)
            npts  = int( npts_v[idx] )
            endidx = startidx+npts
            #print("numpoints for img[{}]: {}".format(idx,npts))
            ssnetout_wcoords = np.zeros( (ssnetout_np.shape[0],ssnetout_np.shape[1]+2), dtype=np.float32 )
            
            ssnetout_wcoords[:,0] = nrows-1-batch_np[startidx:endidx,0] # tick
            ssnetout_wcoords[:,1] = batch_np[startidx:endidx,1] # wire
            
            # pixel value
            ssnetout_wcoords[:,2:2+ssnetout_np.shape[1]] = ssnetout_np[:,:]
            startidx = endidx
            #print("ssnetout_wcoords: {}".format(ssnetout_wcoords[:,0:2]))
            
            meta_v = std.vector("larcv::ImageMeta")()
            for i in range(5):
                meta_v.push_back(meta)
            
            #ssnetout_spimg = larcv.sparseimg_from_ndarray( ssnetout_wcoords, meta_v, larcv.msg.kDEBUG )
            ssnetout_spimg = larcv.sparseimg_from_ndarray( ssnetout_wcoords, meta_v, 0 )            
            bson = larcv.json.as_bson_pybytes( ssnetout_spimg, rseid[0], rseid[1], rseid[2], rseid[3] )
                                          
            bson_reply.append(bson)
    except:
        print("[SparseSSNet] error packing up data for return: {}".format(sys.exc_info()[0]))
        print("[SparseSSNet] trackback")
        print(traceback.format_exc())
        return None
        

    return bson_reply
    
    
    
        
if __name__ == "__main__":

    print("Inference for Sparse-UResNet Semantic Segmentation Network")
    import ROOT 
    from ROOT import std
    from larlite import larlite
    from larcv import larcv
    larcv.PSet
    larcv.json.load_jsonutils
    from ublarcvapp import ublarcvapp

    parser = argparse.ArgumentParser("run Sparse SSNet")
    parser.add_argument("--weight-dir","-w",required=True,type=str,help="Weight files")
    parser.add_argument("--input-larcv","-i",required=True,type=str,help="LArCV file with ADC images")
    parser.add_argument("--tickbackward","-tb",default=False,action='store_true',help="If flag given, reverse time of input image2d before running")
    parser.add_argument("--output", "-o",required=True,type=str,help="output file name")
    parser.add_argument("--nentries","-n",default=-1,type=int,help="number of entries to run. default=-1 (all entries)")
    parser.add_argument("--adcname",type=str,default="wire",help="change the input ADC tree name (default: \"wire\"")
    args = parser.parse_args( sys.argv[1:] )

    tickdir = larcv.IOManager.kTickForward
    if args.tickbackward:
        tickdir = larcv.IOManager.kTickBackward
    
    supera_file = sys.argv[1]
    io = larcv.IOManager(larcv.IOManager.kREAD,"supera",tickdir)
    io.add_in_file( args.input_larcv )
    if args.tickbackward:
        io.reverse_all_products()
    io.initialize()

    outlcv = larcv.IOManager(larcv.IOManager.kWRITE,"lcvout")
    outlcv.set_out_file( args.output )
    outlcv.initialize()
    
    weight_dir = args.weight_dir
    weights = [ weight_dir+"/Plane0_32_5_weighting.ckpt",
                weight_dir+"/Plane1_32_5_weighting.ckpt",
                weight_dir+"/Plane2_32_5_weighting.ckpt" ]

    nentries = io.get_n_entries()
    if args.nentries>0 and args.nentries<nentries:
        nentries = args.nentries

    for ientry in range(nentries):
        io.read_entry(ientry)

        # Event Image
        ev_img = io.get_data( "image2d", args.adcname )
        img_v  = ev_img.Image2DArray()

        results_v = {}
        input_v = {}
        for p in range(img_v.size()):
            pimg_v = std.vector("larcv::Image2D")()
            pimg_v.push_back( img_v.at(p) )
            meta = pimg_v.at(0).meta()                
            # set nice row and col numbers: need powers of 2 for pooling layers to not cause shifts
            nrows = 1024 # 2^10    
            ncols = 3456 # 2^7*3^3
            print("meta[",p,"] ",meta.dump())
            
            sparseimg_v = []
            thresholds = std.vector("float")(1,10.0)

            sparseimg = larcv.SparseImage( pimg_v, thresholds )
            if sparseimg.len()==0:
                continue
        
            iset = p
            bson = larcv.json.as_bson_pybytes( sparseimg, 
                                               io.event_id().run(),
                                               io.event_id().subrun(),
                                               io.event_id().event(), iset )
            sparseimg_v.append(bson)

            results = forwardpass( p, nrows, ncols, sparseimg_v, weights[p] )
            print("num results: ",len(results))
            results_v[p] = results
            input_v[p] = pimg_v.at(0)

        # unpack bson
        evout_ssnet = outlcv.get_data( "sparseimage", "sparseuresnetout" )
        evout_input = outlcv.get_data( "image2d", "ssnetinput" )
        for p,bson in results_v.items():
            c_run    = c_int()
            c_subrun = c_int()
            c_event  = c_int()
            c_id     = c_int()
            c_run.value    = io.event_id().run()
            c_subrun.value = io.event_id().subrun()
            c_event.value  = io.event_id().event()
            c_id.value     = p
            spimg = larcv.json.sparseimg_from_bson_pybytes( bson[0], c_run, c_subrun, c_event, c_id )
            evout_ssnet.Append( spimg )
            evout_input.Append( input_v[p] )
        print("Number of output ssnet sparseimages made: ",evout_ssnet.SparseImageArray().size())
        outlcv.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
        outlcv.save_entry()

    outlcv.finalize()
    io.finalize()

    #os._exit(0)
