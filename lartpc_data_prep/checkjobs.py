import os,sys
import ROOT as rt

#inputlist = "inputlists/bnb_nu_corsika.txt"
#outfolder = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/bnb_nu_corsika/"

#inputlist = "inputlists/bnb_nue_corsika.txt"
#outfolder = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/bnb_nue_corsika/"

inputlist = "inputlists/bnb_nu_pi0filter_corsika.txt"
outfolder = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_showerfragments/bnb_nu_pi0filter_corsika/"

goodlist_name = "goodlist_"+os.path.basename(inputlist)
badlist_name  = "badlist_"+os.path.basename(inputlist)

# parse input list
with open(inputlist,'r') as finput:
    linput = finput.readlines()

input_dict = {}
idx = 1
for l in linput:
    l = l.strip()
    lbase = os.path.basename(l)

    # get number of entries
    print(l)
    inputentries = 0
    try:
        tfile = rt.TFile( l, "open" )
        larlite_id_tree = tfile.Get("larlite_id_tree")
        inputentries = larlite_id_tree.GetEntries()
        tfile.Close()
    except:
        continue

    input_dict[lbase] = {"index":idx,"numentries":0,"entries":[],"inputentries":inputentries}
    
    idx += 1
    
print("Number of input root files: ",len(input_dict))

# parse good run list
goodfiles = []
if os.path.exists(goodlist_name):
    with open(goodlist_name,'r') as f:
        goodlines = f.readlines()
        for l in goodlines:
            l = l.strip()
            if l!="" and l not in goodfiles:
                goodfiles.append(l)
goodfiles.sort()
print("number of good files: ",len(goodfiles))

poutfiles = os.popen(f"find {outfolder} -type f")
outfiles = poutfiles.readlines()

import pointcept
from pointcept.datasets.lartpc import LArTPCDataset

transform=[]

numgood = 0

fbadlist = open( badlist_name, 'w' )

for outfile in outfiles:
    outfile = outfile.strip()

    outfilebase = os.path.basename(outfile)
    origstem = outfilebase[outfilebase.find("_")+1:outfilebase.rfind("_")]+".root"
    entrynum = int(outfilebase.split("_")[-1].split(".")[0][len("entry"):])

    if origstem not in input_dict:
        print("not found in dict: ",origstem)
        continue
    if origstem in goodfiles:
        continue
    
    print(outfile)
    print(origstem," entry=",entrynum)
    
    with open('temp.txt','w') as f:
        print(outfile,file=f)
    ds = LArTPCDataset(data_list_file="temp.txt",transform=transform)
    data = ds.get_data(0)
    print(data.keys())
    if data['coord'].shape[0]>1000:
        isgood = True
        print("  file is good. coord.shape=",data['coord'].shape)
    else:
        isgood = False

    if isgood:
        input_dict[origstem]['numentries'] += 1
        input_dict[origstem]["entries"].append( entrynum )
        numgood += 1
    else:
        print(outfile.strip(),file=fbadlist)

    # for debug
    #if numgood>=30:
    #    break


rerun_indices_name = "rerunid_"+os.path.basename(inputlist)

fgoodlist      = open(goodlist_name,'w')
for goodfile in goodfiles:
    print(goodfile,file=fgoodlist)

frerun_indices = open(rerun_indices_name,'w')
first_good_append = False
for inputfile in input_dict:
    if inputfile in goodfiles:
        continue
    info = input_dict[inputfile]
    if info['numentries']!=info['inputentries']:
        print(inputfile," is missing entries: ",info['numentries'])
        print(info['index'],file=frerun_indices)
    else:
        if not first_good_append:
            #print("",file=fgoodlist)
            first_good_append = True
        print(inputfile,file=fgoodlist)
    




