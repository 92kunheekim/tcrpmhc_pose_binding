"""Conditional pose VAE: encoder & decoder conditioned on a context vector `cond`
(V-gene and/or peptide embeddings). cond_dim must match the conditioning passed in
(e.g. emb_dim for peptide-only, 3*emb_dim for [Va,Vb,peptide])."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BODIES
from geometry import (real_sph_harm, quat_to_matrix, rot6d_to_matrix, matrix_to_rot6d,
                      geodesic_angle, canonicalize_quat, QuaternionLinear)

class CondBodyVAERaw(nn.Module):
    def __init__(self, cond_dim, latent=4, sh_degree=3, hidden=32, rotation_encoder="6d"):
        super().__init__(); self.sh_degree = sh_degree; sh = (sh_degree+1)**2
        if rotation_encoder == "qnn":
            self.qnn = nn.Sequential(QuaternionLinear(1, 8), nn.ReLU(), QuaternionLinear(8, 8)); rot = 32
        else:
            self.qnn = None; rot = 6
        self.enc = nn.Sequential(nn.Linear(1+sh+rot+cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent); self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent+cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.h_reach = nn.Linear(hidden, 1); self.h_dir = nn.Linear(hidden, 3); self.h_rot = nn.Linear(hidden, 6)
    def encode(self, t, q, cond):
        reach = torch.linalg.norm(t, dim=-1, keepdim=True); d = t/reach.clamp_min(1e-8)
        sh = real_sph_harm(d, self.sh_degree); q = canonicalize_quat(q)
        rot = self.qnn(q) if self.qnn is not None else matrix_to_rot6d(quat_to_matrix(q))
        h = self.enc(torch.cat([reach, sh, rot, cond], -1)); return self.mu(h), self.logvar(h)
    def reparam(self, mu, lv): return mu if not self.training else mu+torch.exp(0.5*lv)*torch.randn_like(lv)
    def decode(self, z, cond):
        h = self.dec(torch.cat([z, cond], -1)); return F.softplus(self.h_reach(h)), F.normalize(self.h_dir(h), dim=-1), rot6d_to_matrix(self.h_rot(h))
    def forward(self, t, q, cond):
        mu, lv = self.encode(t, q, cond); z = self.reparam(mu, lv); r, d, R = self.decode(z, cond); return (r, d, R), mu, lv, z

class ConditionalPoseVAERaw(nn.Module):
    def __init__(self, cond_dim, latent_per_body=4, sh_degree=3, rotation_encoder="6d"):
        super().__init__(); self.bodies = BODIES
        self.vaes = nn.ModuleDict({b: CondBodyVAERaw(cond_dim, latent_per_body, sh_degree, rotation_encoder=rotation_encoder) for b in BODIES})
        self.latent_dim = latent_per_body*len(BODIES)
    def forward(self, x, cond):
        rec, mus, lvs, zs = [], [], [], []
        for i, b in enumerate(self.bodies):
            seg = x[:, i*7:(i+1)*7]; r, mu, lv, z = self.vaes[b](seg[:, 0:3], seg[:, 3:7], cond)
            rec.append(r); mus.append(mu); lvs.append(lv); zs.append(z)
        return rec, torch.cat(mus, 1), torch.cat(lvs, 1), torch.cat(zs, 1)

def cvae_loss_masked(recon, x, mu, logvar, mask, beta=0.1):
    m = mask.bool()
    if m.sum() == 0: return x.new_zeros(())
    rec = x.new_zeros(())
    for i, (rh, dh, Rh) in enumerate(recon):
        seg = x[:, i*7:(i+1)*7]; t, q = seg[:, 0:3], seg[:, 3:7]
        reach = torch.linalg.norm(t, dim=-1, keepdim=True); d = t/reach.clamp_min(1e-8); R = quat_to_matrix(canonicalize_quat(q))
        rec = rec + F.mse_loss(rh[m], reach[m]) + (1-(dh[m]*d[m]).sum(-1)).mean() + geodesic_angle(Rh[m], R[m]).mean()
    kld = -0.5*torch.mean((1+logvar-mu.pow(2)-logvar.exp())[m]); return rec+beta*kld
