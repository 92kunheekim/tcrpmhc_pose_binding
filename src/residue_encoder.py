"""Per-residue groove-frame encoder for peptide / CDR3 residues (structure-augmented tower).

Consumes residue_frame_features.csv (from data/scripts/extract_residue_frame_features.py):
each residue = groove-frame position (along, across, height) + side-chain functional-group
direction (reach + unit dir) + amino-acid identity (Atchley) + pLDDT.

`ResidueSetEncoder` runs a small Transformer over a chain's residues and pools to an embedding.
pLDDT GATES the geometry (low-confidence residues contribute less structural signal) and weights
the pooling -- important because CDR3 is the least reliably predicted region.

Use for: the TCR-independent structural peptide tower, and (not-V-marginalized) per-residue CDR3
channel that complements the body-level flow residual.
"""
import numpy as np
import torch
import torch.nn as nn
from atchley import ATCHLEY

GEOM = ["along", "across", "height", "sc_reach", "dx", "dy", "dz"]   # 7 geometric channels
FEAT_DIM = len(GEOM) + 5 + 1                                          # geom + Atchley(5) + conf(1) = 13


def _atch(a):
    return ATCHLEY.get(a, (0.0,)*5)


def residue_arrays(df, chain, id_order, max_len, pos_scale=10.0):
    """Build (N, max_len, FEAT_DIM) features, (N, max_len) mask, (N, max_len) conf for `chain`,
    aligned to `id_order`. Geometry (position Å + reach Å) is divided by `pos_scale`; direction
    unit-vectors and Atchley left as-is; conf = pLDDT/100. Residues beyond max_len are dropped."""
    sub = df[df.chain == chain]
    groups = {i: s for i, s in sub.groupby("id")}
    N = len(id_order)
    X = np.zeros((N, max_len, FEAT_DIM), "float32")
    M = np.zeros((N, max_len), "float32"); C = np.zeros((N, max_len), "float32")
    for n, cid in enumerate(id_order):
        s = groups.get(cid)
        if s is None: continue
        s = s.sort_values("idx").head(max_len)
        for j, (_, r) in enumerate(s.itertuples() if False else s.iterrows()):
            conf = float(r.plddt)/100.0
            X[n, j, 0:4] = np.array([r.along, r.across, r.height, r.sc_reach], "float32")/pos_scale
            X[n, j, 4:7] = np.array([r.dx, r.dy, r.dz], "float32")          # unit direction (unscaled)
            X[n, j, 7:12] = _atch(r.aa); X[n, j, 12] = conf
            M[n, j] = 1.0; C[n, j] = conf
    return X, M, C


class ResidueSetEncoder(nn.Module):
    """Transformer over a chain's per-residue frame features -> pooled embedding (emb_dim).
    pLDDT gates the geometry channels and weights the masked pooling."""
    def __init__(self, emb_dim=64, n_layers=2, n_heads=4, ff=128, max_len=40, dropout=0.1, gate_geom=True):
        super().__init__(); self.gate_geom = gate_geom
        self.proj = nn.Linear(FEAT_DIM, emb_dim)
        self.pos = nn.Parameter(torch.zeros(max_len, emb_dim)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(emb_dim, n_heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask, conf):
        """x (B,L,FEAT_DIM), mask (B,L) 1=valid, conf (B,L) pLDDT/100."""
        if self.gate_geom:
            g = conf.unsqueeze(-1)
            x = torch.cat([x[..., :7]*g, x[..., 7:]], dim=-1)          # down-weight low-confidence geometry
        h = self.proj(x) + self.pos[:x.size(1)].unsqueeze(0)
        pad = ~mask.bool()
        empty = ~mask.bool().any(1)                                    # rows w/ no valid residues (e.g. missing CDR3)
        if empty.any():
            pad = pad.clone(); pad[empty, 0] = False                  # keep >=1 key so attention is defined (avoids NaN)
        h = torch.nan_to_num(self.enc(h, src_key_padding_mask=pad))
        w = (conf*mask).unsqueeze(-1)                                  # pLDDT-weighted pool; empty rows -> weight 0 -> 0
        return self.out((h*w).sum(1)/w.sum(1).clamp_min(1e-6))
