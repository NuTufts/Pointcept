# Provenance of the Tufts-generated simulation (`ub_on_tufts` samples)

How the corsika + BNB full-MC samples behind the Sonata pre-training (and the
LArFormer training data derived from the same ROOT files) were made. Traced
2026-11-11 from the pre-training config back to the simulation configs.

## Trace

| Step | What |
| --- | --- |
| Pre-training config | `configs/lartpc/sonata_pretrain/pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py` |
| File lists | `pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_{trainsplit,valsplit}.txt` (390,000 / 20,000 events), built with `lartpc/data_prep/training_data/make_combined_hdflist.sh` + `make_hdflist_split.sh` |
| HDF5 (gen-1 prep) | `ub_on_tufts/hdf5/v2_expandedclasses/<sample>/` — **deleted**; only the lists remain |
| dlmerged ROOT | `ub_on_tufts/<sample>/NNN/NNN/dlmerged_<TAG>_fileno*.root`; now on `/cluster/tier2/wongjiradlab/larbys/data/ub_on_tufts/` (login-node only). Input lists: `pointcept_env/pointcept/lartpc_data_prep/inputlists/<sample>[_prod2].txt` (line number == fileno) |
| Simulation | [`NuTufts/uboone_on_tufts`](https://github.com/NuTufts/uboone_on_tufts) `nu_and_cosmics/` (commit `b977b98`), checkout at `/cluster/tufts/wongjiradlabnu/twongj01/uboone_on_tufts` |
| Modified MicroBooNE code | [`NuTufts/ublite`](https://github.com/NuTufts/ublite) tag `v08_00_00_40a_dl_tufts1` (fork of `uboone/ublite`, upstream tag `v08_00_00_40a_dl` + 1 commit) |

The later gen-2 (LANTERN, `hdf5/v3_larmatch`) training data is made from the same
dlmerged ROOT files, so this provenance applies to it too.

## Samples in the pre-training list

| Sample | train-list events | dlmerged files (prod2 list) | events generated/file |
| --- | --- | --- | --- |
| `bnb_nu_corsika` | 87,546 | 5,000 | 20 |
| `bnb_nue_corsika` | 90,757 | 4,999 | 20 |
| `bnb_nu_pi0_corsika` | 90,997 | 4,995 | 20 |
| `bnb_nu_pi0filter_corsika` | 56,526 | 8,313 | 50, pi0-filtered |
| `bnb_nu_chargedpiplus_corsika` | 64,174 | 8,050 | 50, pi+-filtered |

Per-sample gen fcl, run/submit script, generator settings and production dates:
see `nu_and_cosmics/README.md` in `uboone_on_tufts`.

## Software and chain

- `uboonecode v08_00_00_40a_dl -q e17:prof` from cvmfs, in the
  `uboone-devel-sl7` container, with a locally built ublite (mrb dev area
  `uboone_on_tufts/nu_and_cosmics/dev/v08_00_00_40a_dl`, larsoft `v08_05_00_13`).
- Chain per file: GENIE+CORSIKA gen (seed `10000 + fileno`, run = fileno) →
  `wirecell_g4_uboone.fcl` → `wirecell_detsim_uboone.fcl` →
  `reco_uboone_mcc9_8_driver_stage1.fcl` →
  `standard_larcv_uboone_mctruth_prod_on_tufts_wcsimch.fcl` → `hadd` of the larlite
  and larcv outputs into one dlmerged file.
- The ublite patch is what makes the truth labels possible: the larlite `simch`
  stream keeps SimChannels for all three planes and stores each IDE's x,y,z
  (stock ublite keeps only Y-plane channels and drops x,y,z). For the WireCell
  `driftWC:simpleSC` SimChannels only y,z are true deposit coordinates; x is the
  wire-plane position (-0.6..0 cm), so the drift coordinate comes from the TDC
  (checked 2026-11-12, `uboone_on_tufts/wireplane_extract/README.md`). It also lets the
  LiteScanner store simulated CRT hits/tracks when there is no DAQ header.

## Open items

- `bnb_nu_pi0_corsika` was made with an earlier revision of the pi0 fcl that
  lacked `SelectEvents`; it is most likely unfiltered BNB nu in the active TPC.
  Which larcv-stage fcl it used is unconfirmed. Settling either needs one of its
  dlmerged files staged from tier2.
- No output file was opened during this trace (tier2 is not mounted on compute
  nodes); the trace rests on scripts, job logs and file lists.
- Launch parameters (`OFFSET`, `stride`, `--array`) of earlier tranches survive
  only in `uboone_on_tufts/nu_and_cosmics/logs/` and `workdir/` (not in git).
