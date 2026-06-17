# Development Datasets on Laptop

This is a list of datasets on the local machine twongjrad@pop-os used for developing LArFormer training tasks.

## mergedh5 datasets

### BNB nu charged pion + Corsika Sample

These are files that have been processed by the driver script `run_lantern_wconfig.sh` (located in `lartpc_data_prep/lantern_scripts/`) using the config file `bnbnu_chargedpiplus_corsika.conf` (located in `lartpc_data_prep/lantern_scripts/lantern_configs/`).

  /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_chargedpiplus_corsika/

We have a copy of the ROOT source files from which they were generated at:

  /mnt/ddrive/data/ub_on_tufts/root/bnb_nu_chargedpiplus_corsika/


The source files made using the MicroBooNE model of the BNB neutrino flux.
Neutrino interactions were filtered such that all neutrino interactions in the files must have
one charged pion in the final state.

### BNB nu neutral pion + Corsika Sample

We also have analagous files for interactions where at least one neutral pion was created by the neutrino interaction.

  /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/merged_h5

We do not have a copy of ROOT source files for this data set. However, we have prepare flash information files as well.

  /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/flashinfo_h5

The flashinfo files were created by running Step 5 in the lantern data preparation process.
The script for that step is `run_flashinfo_wconfig.sh` (located in `lartpc_data_prep/lantern_scripts/`).

