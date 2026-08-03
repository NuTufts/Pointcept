The goal for this study is to select two photon events and create a pi0 mass peak.

The signal events we want to isolate are neutrino interactions that satisfy the following:
- the interaction vertex occurs within the WireCell FV inside the TPC,
- the interaction produces exactly one primary final state pi0 that decays into two photons that are detectable (see below for definition of detectable)

We are interested in tagging the signal as either charged-current or neutral current.
We will want to split the sample into these two categories.

The photon detectability is defined as using the true visible energy of the photon to be above 20 MeV. We use the true visible energy definition used in the evaluation script in lartpc/larformer_reco/export/compare_to_legacy_ntuple.py.

As for the selection criteria using the reconstructed variables, we select events that:
- has two reconstructed photons above 20 MeV,
- has a vertex inside the WireCell FV.
Like with the signal definition, we split the sample into charged-current and neutral current. This is defined by finding at least one muon created at the vertex.

I would like a plot of the invariant mass of the two photons for selected events, divided into charged-current and neutral current events.  Within the CC or NC plots, for events selected from the simulation sample, we tag events as either CC and NC and whether they fall into the signal or background categories. 

We can use the invariant mass formula: $m_{\gamma\gamma} = \sqrt{2 E_1 E_2 (1 - \cos \theta_{12})}$, where $E_1$ and $E_2$ are the energies of the two photons and $\theta_{12}$ is the angle between them. 

We start with the simulated sample processed to evaluate the reconstruction performance.
We use the output of the export ntuple script in lartpc/larformer_reco/export/export_gen2ntuple.py.

The current ntuple is at lartpc/larformer_reco/output/mcc9_bnbnu_overlay_1500_full/dlgen2_larformer_ntuple_mcc9_bnbnu_overlay_1500_full_67k_pre_llr_attach.root.

To start, use only the neutrino slice stream. Use only confidently attached photons as for determining if the event passes the selection criteria. 

Besides the invariant mass plot, I would also be interested efficiency of finding events as well as a function of the total true visible energy of the two photons.

## Secondary SBND-SPINE Comparison

SBND has a CC 1pi0 selection using the SPINE reconstruction to compare to.
While not on MicroBooNE, this is closer to what I would think is the state-of-the-art 
reconstruction for this channel in current LArTPC experiments.

The signal definition is:

- Flash Matched
    - Interaction is matched to ‘valid_flashmatch’ variable
- Fiducial Volume
    - Require interaction vertex to be at least 20 cm from
    - X,Y detector boundaries and 10 [50] cm from
    - upstream [downstream] Z detector boundaries
- Topology
    - 1 primary muon: muon kinetic energy > 143.425 MeV (50 cm long)
    - 1 primary neutral pion (2 primary photons)
        - photon kinetic energy > 20 MeV
        - diphoton mass < 400 MeV/c2
    - 0 charged pions
        - pion kinetic energy > 25 MeV
    - Inclusive to all other particles

From doc-db: https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=47857&filename=June%2026%20SBND%20Collaboration.pdf

After their selection, they report:
  - 86% purity
  - 65% efficiency
