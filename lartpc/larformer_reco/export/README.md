# gen2ntuple Export (task 2)

Tools to export larformer_reco products into the legacy gen2ntuple flat ROOT
format (see ../gen2ntuple/README.md for the branch dictionary; design decisions
in the repo memory + larformer_reco_output_data_schema.md).

- `extract_truth_sidecar.py` — larlite → H5 truth sidecar per merged_dlreco
  file (GENIE + mcreco truth, SCE-corrected, WC-FV flags; POT). Runs in the
  pointcept container with the ubdl stack sourced. Driven by
  `../slurm/submit_truth_sidecar_shard.sh` over
  `inputlists/dlmerged_scale1500_resolved.txt` (fileno = line number).
- `wirecell_fiducial_volume.cxx` + `lib_wirecell_fiducial_volume.so` — the
  Wire-Cell fiducial-volume test (copied from gen2ntuple/helpers; compile with
  g++ -fPIC -shared inside the container).
