"""Peptide-SPECIFIC (fixed-antigen) models + per-peptide evaluation.

Antigen is constant within a peptide, so there is no peptide/MHC tower; the model
learns from TCR sequence and/or pose. Compares seq, pose-only, and seq+pose, plus
tree-FramePose baselines and permutation controls, under 10x repeated
leakage-controlled 80/20 CV.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CHAINS
from encoders import ChainEncoder
from data import build_tree_features, cluster_tcrs, repeated_splits
from train_utils import (run_repeated, aggregate_oof, cluster_bootstrap, DEVICE)

# ---- models ----
class SeqOnly(nn.Module):
    def __init__(self, emb_dim=64, hidden_head=128, dropout=0.3, chains=CHAINS):
        super().__init__(); self.chains = chains
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.head = nn.Sequential(nn.Linear(emb_dim*len(chains), hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b):
        return self.head(torch.cat([self.encoders[c](b[c]) for c in self.chains], 1)).squeeze(-1)

class PoseOnlyClf(nn.Module):
    def __init__(self, pose_vae, hidden_head=128, dropout=0.3, freeze_pose=True):
        super().__init__(); self.pose_vae = pose_vae
        if freeze_pose:
            for q in self.pose_vae.parameters(): q.requires_grad = False
        self.head = nn.Sequential(nn.Linear(pose_vae.latent_dim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b):
        _, mu, _, _ = self.pose_vae(b["pose"]); return self.head(mu).squeeze(-1)

class TCRPoseFusion(nn.Module):
    def __init__(self, pose_vae, emb_dim=64, hidden_head=128, dropout=0.3, chains=CHAINS, freeze_pose=True):
        super().__init__(); self.chains = chains
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains}); self.pose_vae = pose_vae
        if freeze_pose:
            for p in self.pose_vae.parameters(): p.requires_grad = False
        pdim = self.pose_vae.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pdim))
        self.head = nn.Sequential(nn.Linear(emb_dim*len(chains)+pdim, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, 1))
    def forward(self, b):
        z = torch.cat([self.encoders[c](b[c]) for c in self.chains], 1); _, mu, _, _ = self.pose_vae(b["pose"]); mask = b.get("pose_mask")
        if mask is not None:
            mask = mask.float().unsqueeze(1); mu = mask*mu+(1-mask)*self.null_pose.unsqueeze(0)
        return self.head(torch.cat([z, mu], 1)).squeeze(-1)

def run_per_peptide(pool, trans_scale, vae6, vaeq, warm_start, peptides=None, min_class=5):
    """Per-peptide: seq / pose_only / seq+pose (6d,qnn) + tree baselines + controls."""
    ET = lambda: ExtraTreesClassifier(400, class_weight="balanced", n_jobs=-1, random_state=0)
    TREES = {"FramePose_RF": (lambda: RandomForestClassifier(400, class_weight="balanced", n_jobs=-1, random_state=0), False),
             "FramePose_ExtraTrees": (ET, False),
             "FramePose_HistGBT": (lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, random_state=0), True)}
    peptides = peptides or sorted(pool.peptide.unique())
    rows = []
    for pep in peptides:
        sub = pool[pool.peptide == pep].reset_index(drop=True); y = sub.label.to_numpy().astype(int)
        if min((y == 1).sum(), (y == 0).sum()) < min_class: continue
        n = len(y); prev = float(y.mean()); splits = repeated_splits(sub)
        FAC = {"seq": lambda: warm_start(SeqOnly()),
               "pose_only_6d": lambda: PoseOnlyClf(vae6), "pose_only_qnn": lambda: PoseOnlyClf(vaeq),
               "seq+pose_6d": lambda: warm_start(TCRPoseFusion(pose_vae=vae6)),
               "seq+pose_qnn": lambda: warm_start(TCRPoseFusion(pose_vae=vaeq))}
        for name, fac in FAC.items():
            _, P, _, a, pr = run_repeated(fac, sub, splits, trans_scale)
            rows.append(dict(peptide=pep, model=name, prevalence=round(prev, 4),
                             auroc=round(a.mean(), 4), auroc_sd=round(a.std(), 4), auprc=round(pr.mean(), 4)))
        Xt = build_tree_features(sub)
        for tname, (tfn, sw) in TREES.items():
            a, pr = _tree_repeated(Xt, y, splits, tfn, sw)
            rows.append(dict(peptide=pep, model=tname, prevalence=round(prev, 4),
                             auroc=round(a.mean(), 4), auroc_sd=round(a.std(), 4), auprc=round(pr.mean(), 4)))
    return pd.DataFrame(rows)

def _tree_repeated(X, y, splits, model_fn, sw):
    a = []; p = []
    for tr, te in splits:
        m = model_fn()
        if sw:
            w = np.where(y[tr] == 1, (y[tr] == 0).sum()/max((y[tr] == 1).sum(), 1), 1.0); m.fit(X[tr], y[tr], sample_weight=w)
        else: m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1: a.append(roc_auc_score(y[te], pr)); p.append(average_precision_score(y[te], pr))
    return np.array(a), np.array(p)
