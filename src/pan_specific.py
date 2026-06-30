"""PAN-specific model (FUTURE) — across peptide AND MHC.

Extends the across-peptide model with an MHC tower so the model generalizes over
HLA alleles, not just A*02:01. This is a scaffold/design stub; not yet wired into
the pipeline.

Plan
----
1. MHC encoder. Two options:
   (a) ESM-2 embedding of the full MHC protein sequence (pMTnet-omni style) — pass
       MHC (class I: heavy chain + b2m; class II: alpha+beta) through a frozen ESM-2,
       project to emb_dim. Best generalization across alleles.
   (b) Lightweight: ChainEncoder over an Atchley-encoded MHC pseudo-sequence
       (contact residues), if ESM-2 is too heavy.
2. Conditioning. The pose is relative to the pMHC groove, so for a pan model the
   pose conditioning should include the MHC embedding too:
       cond = [Va_emb, Vb_emb, peptide_emb, mhc_emb].
   Set ConditionalPoseVAERaw(cond_dim = 3*emb_dim + mhc_dim) accordingly.
3. Fusion head sees [t, peptide_emb, mhc_emb, t*peptide, muf] (+ interactions).
4. Negatives / specificity: mismatch over BOTH peptide and MHC (a binder is a
   negative for non-cognate peptide-MHC). Keep posed-only for pose discrimination.
5. Evaluation: leave-one-allele-out and leave-one-peptide-out for true pan tests.

Skeleton below mirrors CondPeptideAwareFusion with an added MHC tower; fill in the
MHC encoder and feed mhc into `cond` and the head, then register in an AFS dict.
"""
import torch
import torch.nn as nn
from config import CHAINS, EMB_DIM
from encoders import ChainEncoder
from pose_cvae import ConditionalPoseVAERaw

class MHCEncoder(nn.Module):
    """Placeholder. Replace with ESM-2 (frozen) -> Linear(emb_dim), or an Atchley
    ChainEncoder over the MHC pseudo-sequence."""
    def __init__(self, emb_dim=EMB_DIM, mhc_max_len=380):
        super().__init__()
        self.encoder = ChainEncoder(emb_dim=emb_dim)   # swap for ESM-2 projection later
        self.mhc_max_len = mhc_max_len
    def forward(self, mhc_atchley):                    # (B, L, 5)
        return self.encoder(mhc_atchley)

class PanFusion(nn.Module):
    """seq + peptide + MHC + MHC/peptide/V-conditioned pose. NOT yet trained/validated."""
    def __init__(self, emb_dim=EMB_DIM, hidden_head=128, dropout=0.3, chains=CHAINS, rotation_encoder="6d"):
        super().__init__(); self.chains = chains
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.pep_encoder = ChainEncoder(emb_dim=emb_dim)
        self.mhc_encoder = MHCEncoder(emb_dim=emb_dim)
        self.tcr_proj = nn.Linear(emb_dim*len(chains), emb_dim)
        self.cpose = ConditionalPoseVAERaw(cond_dim=4*emb_dim, rotation_encoder=rotation_encoder)  # [Va,Vb,pep,mhc]
        pdim = self.cpose.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pdim))
        self.head = nn.Sequential(nn.Linear(emb_dim*4+pdim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b, return_aux=False):
        enc = {c: self.encoders[c](b[c]) for c in self.chains}
        t = self.tcr_proj(torch.cat([enc[c] for c in self.chains], 1))
        p = self.pep_encoder(b["pep"]); h = self.mhc_encoder(b["mhc"])
        cond = torch.cat([enc["va"], enc["vb"], p, h], dim=1)
        recon, mu, lv, _ = self.cpose(b["pose"], cond); mask = b.get("pose_mask"); muf = mu
        if mask is not None:
            mk = mask.float().unsqueeze(1); muf = mk*mu+(1-mk)*self.null_pose.unsqueeze(0)
        logit = self.head(torch.cat([t, p, h, t*p, muf], dim=1)).squeeze(-1)
        return (logit, recon, mu, lv) if return_aux else logit
