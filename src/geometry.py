"""Rotation/translation geometry: spherical harmonics (S2), quaternion<->matrix,
6D rotation rep, SO(3) geodesic, quaternion-NN layer."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def real_sph_harm(d, L=3):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    f = [torch.full_like(x, 0.282095), 0.488603*y, 0.488603*z, 0.488603*x,
         1.092548*x*y, 1.092548*y*z, 0.315392*(3*z*z-1), 1.092548*x*z, 0.546274*(x*x-y*y),
         0.590044*y*(3*x*x-y*y), 2.890611*x*y*z, 0.457046*y*(5*z*z-1), 0.373176*z*(5*z*z-3),
         0.457046*x*(5*z*z-1), 1.445306*z*(x*x-y*y), 0.590044*x*(x*x-3*y*y)]
    return torch.stack(f, -1)

def quat_to_matrix(q):
    q = F.normalize(q, dim=-1); w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)
    return R.reshape(*q.shape[:-1], 3, 3)

def rot6d_to_matrix(r6):
    a1, a2 = r6[..., 0:3], r6[..., 3:6]; b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1*a2).sum(-1, keepdim=True)*b1; b2 = F.normalize(a2, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)

def matrix_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)

def geodesic_angle(Ra, Rb, eps=1e-6):
    Rd = torch.matmul(Ra.transpose(-1, -2), Rb); tr = Rd[..., 0, 0]+Rd[..., 1, 1]+Rd[..., 2, 2]
    return torch.arccos(torch.clamp((tr-1)*0.5, -1+eps, 1-eps))

def canonicalize_quat(q):
    q = F.normalize(q, dim=-1); return torch.where(q[..., :1] < 0, -q, q)

class QuaternionLinear(nn.Module):
    """Hamilton-product linear layer; in/out count quaternions, tensor carries 4x reals [r|i|j|k]."""
    def __init__(self, in_q, out_q):
        super().__init__(); k = 1.0/math.sqrt(in_q)
        self.r = nn.Parameter(torch.empty(out_q, in_q).uniform_(-k, k))
        self.i = nn.Parameter(torch.empty(out_q, in_q).uniform_(-k, k))
        self.j = nn.Parameter(torch.empty(out_q, in_q).uniform_(-k, k))
        self.k = nn.Parameter(torch.empty(out_q, in_q).uniform_(-k, k))
    def forward(self, x):
        xr, xi, xj, xk = x.chunk(4, dim=-1)
        o_r = F.linear(xr, self.r)-F.linear(xi, self.i)-F.linear(xj, self.j)-F.linear(xk, self.k)
        o_i = F.linear(xr, self.i)+F.linear(xi, self.r)+F.linear(xj, self.k)-F.linear(xk, self.j)
        o_j = F.linear(xr, self.j)-F.linear(xi, self.k)+F.linear(xj, self.r)+F.linear(xk, self.i)
        o_k = F.linear(xr, self.k)+F.linear(xi, self.j)-F.linear(xj, self.i)+F.linear(xk, self.r)
        return torch.cat([o_r, o_i, o_j, o_k], dim=-1)
