import os,sys
import pointcept
import numpy as np

from pointcept.datasets import LArTPCDataset, LArTPCInstanceDataset
from pointcept.datasets.transform import RandomFlipAxis

from larlite import larutil
geom = larutil.Geometry.GetME()
print(geom)

firstwireproj = geom.GetFirstWireProj()
xorthz = geom.GetOrthVectorsZ()
xorthy = geom.GetOrthVectorsY()

orthz = (xorthz[0],xorthz[1],xorthz[2])
orthy = (xorthy[0],xorthy[1],xorthy[2])

print("firstwireproj: ",firstwireproj[0],", ",firstwireproj[1],", ",firstwireproj[2])
print("orthz: ",orthz)
print("orthy: ",orthy)

import ROOT as rt

wire_projections = [
    ((0.0,    0.0,  -338.6334821387676), (0.0, -0.866, 0.5)),
    ((0.0,    0.0,  -333.0331845276306), (0.0,  0.866, 0.5)),
    ((0.0,    0.0,  0.33), (0.0, 0.0, 1.0))
]

wire_scale = 1.0/3456.0

transform = [
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="Copy", keys_dict={"color": "origin_color"}),
    dict(type="Copy", keys_dict={"strength": "origin_strength"}),
    #dict(type="RandomFlipAxis",p=1.0,axis='z',center=0.5*1036.0,wire_projections=wire_projections,coord_scale=1.0,swap_strength_columns=(0, 1)),
    #dict(type="RandomFlipAxis",p=1.0,axis='z',center='mean',wire_projections=wire_projections,coord_scale=1.0,swap_strength_columns=(0, 1)),
    dict(type="RandomFlipAxis",p=1.0,axis='y',center=0.0,wire_projections=wire_projections,coord_scale=1.0,swap_strength_columns=(0, 1)),
    #dict(type="RandomFlipAxis",p=1.0,axis='y',center='mean',wire_projections=wire_projections,coord_scale=1.0,swap_strength_columns=(0, 1)),
    dict(
        type="Collect",
        keys=(
            "origin_color",
            "origin_coord",
            "color",
            "coord",
            "strength",
            "origin_strength"
        ),
        offset_keys_dict=dict(),
        # Features: strength (energy) + color (wire coords)
        #global_feat_keys=("global_strength", "global_color"),
        #local_feat_keys=("local_strength", "local_color"),
    ),

]

x = LArTPCDataset(coord_scale=1.0,
                  #data_root="data/lartpc",
                  data_list_file="train_split.txt",
                  transform=transform,
                  exclude_other=True,
                  include_ghosts=True)

data = x[0]
print(data.keys())
print("coord: ")
print(data['coord'][:10,:])

print("original coord: ")
print(data['origin_coord'][:10,:])

print("Compare 'color' i.e. Flipped Wire Coordinates")
for i in range(5):
    print("=="*10)
    orig_wires = data['origin_color'][i,:]
    flip_wires = data['color'][i,:]
    flip_pos   = data['coord'][i,:]
    pix        = data['strength'][i,:]
    orig_pix   = data['origin_strength'][i,:]
    print(f"[{i}] original=",orig_wires,"  <--> flipped=",flip_wires)

    wire_check = np.zeros(3,dtype=np.float32)
    vpos = rt.TVector3()
    for v in range(3):
        vpos[v] = flip_pos[v]
    #print(" flip tvector3=",vpos)
    for p in range(3):
        wire_check[p] = geom.WireCoordinate( vpos, p )*wire_scale
    print("  check flipped wire coordinates with larutil: ",wire_check)

    print(" orig pixval=",orig_pix,"  <-->  ",pix)



