~/.local/bin/i3cols extr_sep  ~/nn_input_link/l7/120000/i3/oscNext_genie_level7_v01.04_pass2.120000.000* \
    --outdir  ~/nn_input_link/l7/120000/i3cols/ \
    --procs 50 \
    --concatenate-and-index-by subrun \
    --keys I3EventHeader \
           I3MCTree \
           MCInIcePrimary \
           InIcePulses \
           SplitInIcePulsesSRT \
           SplitInIcePulsesTWSRT \
           SRTTWOfflinePulsesDC \
           I3MCWeightDict \
           MCDeepCoreStartingEvent \
           I3MCWeightDict \
           L7_MuonClassifier_ProbNu \
           L7_CoincidentMuon_bool \
           L7_oscNext_bool \
           L7_reconstructed_zenith \
           L7_reconstructed_total_energy \
           L7_PIDClassifier_ProbTrack
