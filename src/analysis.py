"""Latent-space / reconstruction analysis for comparing two pose VAEs
(e.g. binder-only vs non-binder-only)."""
import numpy as np
import torch
import torch.nn.functional as F
from geometry import quat_to_matrix, canonicalize_quat, geodesic_angle

@torch.no_grad()
def recon_error_per_sample(vae, X, device, bs=512):
    """Per-structure reconstruction error: sum over bodies of
    reach-MSE + S2 geodesic(direction) + SO(3) geodesic(rotation). No KL. Returns (N,)."""
    vae.eval(); errs = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)
        recon, mu, lv, z = vae(x); e = torch.zeros(x.size(0), device=device)
        for bi, (rh, dh, Rh) in enumerate(recon):
            seg = x[:, bi*7:(bi+1)*7]; t, q = seg[:, 0:3], seg[:, 3:7]
            reach = torch.linalg.norm(t, dim=-1, keepdim=True); d = t/reach.clamp_min(1e-8)
            R = quat_to_matrix(canonicalize_quat(q))
            e = e + F.mse_loss(rh, reach, reduction="none").squeeze(-1) + (1-(dh*d).sum(-1)) + geodesic_angle(Rh, R)
        errs.append(e.cpu().numpy())
    return np.concatenate(errs)

@torch.no_grad()
def latent_mu(vae, X, device, bs=512):
    """Deterministic latent embedding mu, shape (N, latent_dim)."""
    vae.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device); _, mu, _, _ = vae(x); out.append(mu.cpu().numpy())
    return np.concatenate(out)

def linear_cka(X, Y):
    """Linear Centered Kernel Alignment between two (N,d) representations. 1 = identical."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro")**2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12
    return float(hsic/denom)
