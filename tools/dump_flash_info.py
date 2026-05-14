from larlite import larlite


infile = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/tmp_workdir/lantern_bnb_nue_corsika_jobid0000_line00426/merged_dlreco_with_ssnet.root"
io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename(infile)
io.open()

flashtree = "simpleFlashBeam"
#flashtree = "simpleFlashCosmic"

for i in range(io.get_entries()):
    io.go_to(i)
    flash_v = io.get_data(larlite.data.kOpFlash, flashtree)
    print("Entry {}: flash_v.size() = {}".format(i, flash_v.size()))
    for j in range(flash_v.size()):
        flash = flash_v[j]
        nch = flash.nOpDets()
        pe = 0
        for k in range(nch):
            pe += flash.PE(k)
        print("  Flash {}: t={}, PE = {}".format(j, flash.Time(), pe))

io.close()