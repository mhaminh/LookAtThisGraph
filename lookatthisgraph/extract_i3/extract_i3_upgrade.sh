# ~/.local/bin/i3cols extr_sep /remote/ceph/user/h/haminh/GraphNet/input/upgrade/1657/i3/* \
~/.local/bin/i3cols extr_as_one /remote/ceph/user/h/haminh/GraphNet/input/upgrade/1657/i3/* \
    --outdir /remote/ceph/user/h/haminh/GraphNet/input/upgrade/1657/i3cols/ \
    --procs 20 \
    --keys I3EventHeader \
           I3MCTree \
           MCInIcePrimary \
           InIcePulses \
           SplitInIcePulsesSRT \
           SplitInIcePulsesTWSRT \
           TimeShift
    # --concatenate-and-index-by subrun \
