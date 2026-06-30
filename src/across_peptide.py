"""ACROSS-peptide models + evaluation (peptide tower; MHC still fixed).

Towers: TCR (4 chain encoders), peptide encoder, and a pose VAE that is either
unconditioned (PoseVAERaw) or conditioned on reused embeddings (ConditionalPoseVAERaw).
Conditioning options: peptide-only (cond_dim=emb_dim) or [Va,Vb,peptide] (cond_dim=3*emb_dim).

Evaluation: MIXED (10x repeated leakage-controlled 80/20, novel TCR) and LOPO
(leave-one-peptide-out, unseen epitope). Pose-discrimination uses posed examples
only (mismatch negatives off) to avoid the null-pose/label confound.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CHAINS, EMB_DIM
from encoders import ChainEncoder
from pose_cvae import ConditionalPoseVAERaw
from data import PairData, cluster_tcrs, repeated_splits, add_mismatch
from train_utils import (run_repeated, train_one, train_one_aux, predict,
                         aggregate_oof, cluster_bootstrap, DEVICE)
from torch.utils.data import DataLoader

class PeptideAwareFusion(nn.Module):
    """seq + peptide (+ optional unconditioned pose). pose_vae=None -> seq+pep."""
    def __init__(self, pose_vae=None, emb_dim=EMB_DIM, hidden_head=128, dropout=0.3, chains=CHAINS, freeze_pose=True):
        super().__init__(); self.chains = chains
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.pep_encoder = ChainEncoder(emb_dim=emb_dim); self.tcr_proj = nn.Linear(emb_dim*len(chains), emb_dim)
        self.pose_vae = pose_vae; pose_dim = 0
        if pose_vae is not None:
            if freeze_pose:
                for p in self.pose_vae.parameters(): p.requires_grad = False
            pose_dim = pose_vae.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pose_dim))
        self.head = nn.Sequential(nn.Linear(emb_dim*3+pose_dim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b):
        t = self.tcr_proj(torch.cat([self.encoders[c](b[c]) for c in self.chains], 1)); p = self.pep_encoder(b["pep"]); feats = [t, p, t*p]
        if self.pose_vae is not None:
            _, mu, _, _ = self.pose_vae(b["pose"]); mask = b.get("pose_mask")
            if mask is not None:
                mask = mask.float().unsqueeze(1); mu = mask*mu+(1-mask)*self.null_pose.unsqueeze(0)
            feats.append(mu)
        return self.head(torch.cat(feats, 1)).squeeze(-1)

def _cond(enc, p, cond_on):
    if cond_on == "peptide": return p
    return torch.cat([enc["va"], enc["vb"], p], dim=1)   # "vpep"

class CondPeptideAware(nn.Module):
    """Conditional pose-only (head sees pose latent only). cond_on in {'peptide','vpep'}."""
    def __init__(self, emb_dim=EMB_DIM, hidden_head=128, dropout=0.3, chains=CHAINS, rotation_encoder="6d", cond_on="vpep"):
        super().__init__(); self.chains = chains; self.cond_on = cond_on
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.pep_encoder = ChainEncoder(emb_dim=emb_dim)
        cdim = emb_dim if cond_on == "peptide" else 3*emb_dim
        self.cpose = ConditionalPoseVAERaw(cond_dim=cdim, rotation_encoder=rotation_encoder)
        pdim = self.cpose.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pdim))
        self.head = nn.Sequential(nn.Linear(pdim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b, return_aux=False):
        enc = {c: self.encoders[c](b[c]) for c in self.chains}; p = self.pep_encoder(b["pep"])
        cond = _cond(enc, p, self.cond_on)
        recon, mu, lv, _ = self.cpose(b["pose"], cond); mask = b.get("pose_mask"); muf = mu
        if mask is not None:
            mk = mask.float().unsqueeze(1); muf = mk*mu+(1-mk)*self.null_pose.unsqueeze(0)
        logit = self.head(muf).squeeze(-1)
        return (logit, recon, mu, lv) if return_aux else logit

class CondPeptideAwareFusion(nn.Module):
    """Full fusion: seq + peptide + conditional pose latent. cond_on in {'peptide','vpep'}."""
    def __init__(self, emb_dim=EMB_DIM, hidden_head=128, dropout=0.3, chains=CHAINS, rotation_encoder="6d", cond_on="vpep"):
        super().__init__(); self.chains = chains; self.cond_on = cond_on
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.pep_encoder = ChainEncoder(emb_dim=emb_dim); self.tcr_proj = nn.Linear(emb_dim*len(chains), emb_dim)
        cdim = emb_dim if cond_on == "peptide" else 3*emb_dim
        self.cpose = ConditionalPoseVAERaw(cond_dim=cdim, rotation_encoder=rotation_encoder)
        pdim = self.cpose.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pdim))
        self.head = nn.Sequential(nn.Linear(emb_dim*3+pdim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b, return_aux=False):
        enc = {c: self.encoders[c](b[c]) for c in self.chains}
        t = self.tcr_proj(torch.cat([enc[c] for c in self.chains], 1)); p = self.pep_encoder(b["pep"])
        cond = _cond(enc, p, self.cond_on)
        recon, mu, lv, _ = self.cpose(b["pose"], cond); mask = b.get("pose_mask"); muf = mu
        if mask is not None:
            mk = mask.float().unsqueeze(1); muf = mk*mu+(1-mk)*self.null_pose.unsqueeze(0)
        logit = self.head(torch.cat([t, p, t*p, muf], dim=1)).squeeze(-1)
        return (logit, recon, mu, lv) if return_aux else logit

def run_mixed(pool, trans_scale, AFS, AUXSET=(), use_mismatch=False):
    n = len(pool); groups = cluster_tcrs(pool); yM = pool.label.to_numpy().astype(int)
    splits = repeated_splits(pool); oofs = {}; rows = []
    for name, fac in AFS.items():
        _, Pc, IDX, a, pr = run_repeated(fac, pool, splits, trans_scale, augment=use_mismatch, aux=(name in AUXSET))
        pagg, tested = aggregate_oof(IDX, Pc, n); oofs[name] = (pagg, tested)
        for i in range(len(a)):
            rows.append(dict(regime="MIXED", split=f"rep{i}", model=name, auroc=round(float(a[i]), 4), auprc=round(float(pr[i]), 4)))
    return pd.DataFrame(rows), oofs, yM, groups

def run_lopo(pool, trans_scale, AFS, AUXSET=(), use_mismatch=False, epochs=15):
    rows = []
    for held in sorted(pool.peptide.unique()):
        te = pool[pool.peptide == held]; tr = pool[pool.peptide != held]
        if min((te.label == 1).sum(), (te.label == 0).sum()) < 3: continue
        trf = add_mismatch(tr) if use_mismatch else tr
        for name, fac in AFS.items():
            model = fac().to(DEVICE)
            tl = DataLoader(PairData(trf.reset_index(drop=True), trans_scale), batch_size=64, shuffle=True)
            vl = DataLoader(PairData(te.reset_index(drop=True), trans_scale), batch_size=128)
            yt = trf.label.to_numpy().astype(int); pw = (yt == 0).sum()/max((yt == 1).sum(), 1)
            (train_one_aux if name in AUXSET else train_one)(model, tl, epochs, 1e-3, pw); pr, yy = predict(model, vl)
            rows.append(dict(regime="LOPO", split=held, model=name, prevalence=round(float(te.label.mean()), 4),
                             auroc=round(roc_auc_score(yy, pr), 4), auprc=round(average_precision_score(yy, pr), 4)))
    return pd.DataFrame(rows)
