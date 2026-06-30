"""Data loading, V-gene/CDR3 extraction, leakage-controlled clustering, splits,
PairData (chains + peptide + raw pose + mask), and tree-feature builder."""
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from atchley import encode_sequence
from config import BODIES, CHAINS, CHAIN_MAXLEN, PEP_MAXLEN, raw_columns

RAW = raw_columns()
_PAT = re.compile(r"C[A-Z]{4,28}?[FW]G[A-Z]G")

def split_chain(chain):
    """(cdr3_core, V-region prefix) from a full TCR chain via the IMGT junction."""
    if not isinstance(chain, str): return "", ""
    h = list(_PAT.finditer(chain))
    if not h: return "", chain[:110]
    m = h[-1]; return m.group()[1:-4], chain[:m.start()]

def cluster_tcrs(sub, ident=0.8):
    """Single-linkage clusters: same V gene (both chains) + >=ident CDR3 identity (both)."""
    n = len(sub); ca, va, cb, vb = [], [], [], []
    for s in sub.tcra_seq:
        c, v = split_chain(s); ca.append(c); va.append(v)
    for s in sub.tcrb_seq:
        c, v = split_chain(s); cb.append(c); vb.append(v)
    par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def idn(a, b): return 0.0 if (len(a) != len(b) or not a) else sum(p == q for p, q in zip(a, b))/len(a)
    bl = defaultdict(list)
    for i in range(n): bl[(va[i], vb[i], len(ca[i]), len(cb[i]))].append(i)
    for mem in bl.values():
        for a in range(len(mem)):
            for b in range(a+1, len(mem)):
                i, j = mem[a], mem[b]
                if idn(cb[i], cb[j]) >= ident and idn(ca[i], ca[j]) >= ident: par[find(j)] = find(i)
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1]

def load_data(data_dir, min_iptm=0.5):
    """Returns dict with pool df (ipTM filtered, posed), trans_scale, and POSE_ALL pretrain corpus."""
    seq = pd.read_csv(f"{data_dir}/combined_sorted.csv")
    seq["cdr3a"], seq["va"] = zip(*seq.tcra_seq.map(split_chain))
    seq["cdr3b"], seq["vb"] = zip(*seq.tcrb_seq.map(split_chain))
    desc = pd.read_csv(f"{data_dir}/pose_descriptors.csv").dropna(subset=RAW)
    ip = pd.read_csv(f"{data_dir}/iptm_table.csv")
    allraw = desc[RAW].to_numpy("float32")
    trans_scale = float(np.mean(np.concatenate([np.linalg.norm(allraw[:, b*7:b*7+3], axis=1) for b in range(len(BODIES))])))
    POSE_ALL = _scale_pose(desc[RAW].to_numpy("float32"), trans_scale)
    m = seq.merge(desc[["id"]+RAW], on="id").merge(ip, on="id")
    pool = m[m.tcr_pmhc_iptm >= min_iptm].reset_index(drop=True)
    pool["pep"] = pool["peptide"]; pool["pose_mask"] = 1.0
    return dict(pool=pool, seq=seq, trans_scale=trans_scale, POSE_ALL=POSE_ALL, RAW=RAW)

def _scale_pose(A, trans_scale):
    A = A.copy()
    for b in range(len(BODIES)): A[:, b*7:b*7+3] /= trans_scale
    return A

def build_pose(frame, trans_scale):
    return _scale_pose(frame[RAW].to_numpy("float32"), trans_scale)

def _quat6d(q):
    q = q/(np.linalg.norm(q, axis=-1, keepdims=True)+1e-9); q = np.where(q[..., :1] < 0, -q, q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = np.stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)], -1)
    c1 = np.stack([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)], -1)
    return np.concatenate([c0, c1], -1)

def build_tree_features(frame):
    """Tree-friendly per-body features: reach + unit-direction + 6D rotation (10/body)."""
    A = frame[RAW].to_numpy("float32"); feats = []
    for b in range(len(BODIES)):
        seg = A[:, b*7:(b+1)*7]; t = seg[:, 0:3]; q = seg[:, 3:7]
        reach = np.linalg.norm(t, axis=1, keepdims=True); d = t/(reach+1e-9)
        feats.append(np.concatenate([reach, d, _quat6d(q)], axis=1))
    return np.concatenate(feats, axis=1).astype("float32")

def add_mismatch(tr, seed=0):
    """Specificity negatives (peptide swapped, no structure). Use ONLY for sequence/peptide
    studies; mixing with posed examples confounds pose models (null-pose=label proxy)."""
    rng = np.random.default_rng(seed); pos = tr[tr.label == 1]; peps = tr.peptide.unique(); rows = []
    for _, r in pos.iterrows():
        others = [p for p in peps if p != r.peptide]
        if not others: continue
        nr = r.copy(); q = rng.choice(others); nr["peptide"] = q; nr["pep"] = q; nr["label"] = 0; nr["pose_mask"] = 0.0
        for c in RAW: nr[c] = 0.0
        rows.append(nr)
    return pd.concat([tr, pd.DataFrame(rows)], ignore_index=True) if rows else tr

def repeated_splits(frame, n_repeats=10):
    """10 independent 80/20 grouped+stratified holdouts (whole V-gene/CDR3 clusters held out)."""
    g = cluster_tcrs(frame); y = frame.label.to_numpy().astype(int); out = []
    for r in range(n_repeats):
        tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=100+r).split(np.zeros(len(y)), y, g)))
        out.append((tr, te))
    return out

class PairData(Dataset):
    def __init__(self, frame, trans_scale):
        self.enc = {c: np.stack([encode_sequence(s, CHAIN_MAXLEN[c]) for s in frame[c]]).astype("float32") for c in CHAINS}
        self.pep = np.stack([encode_sequence(s, PEP_MAXLEN) for s in frame["pep"]]).astype("float32")
        self.pose = build_pose(frame, trans_scale); self.mask = frame["pose_mask"].to_numpy("float32")
        self.y = frame["label"].to_numpy("float32")
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        it = {c: torch.from_numpy(self.enc[c][i]) for c in CHAINS}
        it["pep"] = torch.from_numpy(self.pep[i]); it["pose"] = torch.from_numpy(self.pose[i]); it["pose_mask"] = torch.tensor(self.mask[i])
        return it, torch.tensor(self.y[i])
