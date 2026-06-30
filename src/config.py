"""Shared configuration: body set, sequence chains, padding lengths."""
# 7 FramePose bodies: whole TCR + CDR1/2/3 of alpha and beta
BODIES = ("TCR", "cdr1a", "cdr2a", "cdr3a", "cdr1b", "cdr2b", "cdr3b")
# TCR sequence chains encoded by the sequence tower
CHAINS = ("va", "vb", "cdr3a", "cdr3b")
CHAIN_MAXLEN = {"va": 110, "vb": 110, "cdr3a": 25, "cdr3b": 25}
PEP_MAXLEN = 12
EMB_DIM = 64

def raw_columns(bodies=BODIES):
    """pose_descriptors.csv columns, body-major: [tx,ty,tz, qw,qx,qy,qz] per body."""
    cols = []
    for b in bodies:
        cols += [f"{b}_dx_mhc", f"{b}_dy_mhc", f"{b}_dz_mhc",
                 f"{b}_quat_w", f"{b}_quat_x", f"{b}_quat_y", f"{b}_quat_z"]
    return cols
