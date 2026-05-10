import os
import argparse

import ROOT as rt

from larlite import larlite
from larcv import larcv


parser = argparse.ArgumentParser("2D larcv event display script")
parser.add_argument("-i", "--infile", type=str, required=True, help="input larcv images file")
parser.add_argument("-o", "--outfile", type=str, required=True, help="output larcv images file")
parser.add_argument("-tb", "--tickbackward", default=False, action="store_true", help="store output images in time reverse order")
parser.add_argument("-n","--nentries",default=-1,type=int,help="number of entries to run. default=-1 (all entries)")
parser.add_argument("--adcname",default="wire",type=str,help="Set tree name from which to get ADC wire plane images")
args = parser.parse_args()

iolcv = larcv.IOManager(larcv.IOManager.kREAD, "IOManager_In", larcv.IOManager.kTickForward)
iolcv.add_in_file(args.infile)
#iolcv.reverse_all_products()
iolcv.initialize()

tickdir = larcv.IOManager.kTickForward
if args.tickbackward:
    tickdir = larcv.IOManager.kTickBackward
iolcv_out = larcv.IOManager(larcv.IOManager.kWRITE, "IOManager_Out", tickdir)
iolcv_out.set_out_file(args.outfile)
iolcv_out.initialize()


def reverse_row(row):
  return abs(row-1007)

nentries = iolcv.get_n_entries()
if args.nentries>0 and args.nentries<nentries:
    nentries = args.nentries


for i in range(nentries):

  iolcv.read_entry(i)
  evt_wire = iolcv.get_data(larcv.kProductImage2D, args.adcname)
  adc_wire_v = evt_wire.Image2DArray()
  sparseimg_v = iolcv.get_data(larcv.kProductSparseImage, "sparseuresnetout").SparseImageArray()
  #sparseimg_v = iolcv.get_data(larcv.kProductSparseImage, "sparsessnet").SparseImageArray()
  #evtimg_ubspurn_recs = [larcv.EventImage2D(), larcv.EventImage2D(), larcv.EventImage2D()]
  evtimg_ubspurn_recs = [iolcv_out.get_data(larcv.kProductImage2D, "ubspurn_plane0"),
                         iolcv_out.get_data(larcv.kProductImage2D, "ubspurn_plane1"),
                         iolcv_out.get_data(larcv.kProductImage2D, "ubspurn_plane2")]

  if args.adcname=="wiremc":
    evadc_wirecopy = iolcv_out.get_data(larcv.kProductImage2D,"wire")

  for p in range(3):

    evtimg_ubspurn_recs[p].clear()

    meta = adc_wire_v[p].meta()
    sparsemeta = sparseimg_v[p].meta(0)
    shower = larcv.Image2D(meta)
    track = larcv.Image2D(meta)
    shower.paint(0)
    track.paint(0)
    stride = sparseimg_v[p].nfeatures()+2
    npts = sparseimg_v[p].len()

    if args.adcname=="wiremc":
      wirecopy = adc_wire_v[p]
      evadc_wirecopy.Append(wirecopy)

    for ipt in range(npts):
      row = int(sparseimg_v[p].pixellist().at(ipt*stride+0))
      col = int(sparseimg_v[p].pixellist().at(ipt*stride+1))
      #xrow = meta.row( sparsemeta.pos_y( row ) )
      #xcol = meta.col( sparsemeta.pos_x( col ) )
      xrow = row
      if args.tickbackward:
        xrow = reverse_row(row)
      xcol = col
      hip = sparseimg_v[p].pixellist().at(ipt*stride+2)
      mip = sparseimg_v[p].pixellist().at(ipt*stride+3)
      shr = sparseimg_v[p].pixellist().at(ipt*stride+4)
      dlt = sparseimg_v[p].pixellist().at(ipt*stride+5)
      mic = sparseimg_v[p].pixellist().at(ipt*stride+6)
      shower_score = shr+dlt+mic
      track_score  = hip+mip
      shower.set_pixel(xrow, xcol, shower_score)
      track.set_pixel(xrow, xcol, track_score)

    evtimg_ubspurn_recs[p].Append(shower)
    evtimg_ubspurn_recs[p].Append(track)

  iolcv_out.set_id(evt_wire.run(), evt_wire.subrun(), evt_wire.event())
  iolcv_out.save_entry()

#iolcv_out.reverse_all_products()
iolcv_out.finalize()

#os._exit(0)
