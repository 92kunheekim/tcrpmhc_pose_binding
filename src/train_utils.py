"""Training, pretraining, repeated leakage-controlled CV, and cluster bootstrap."""
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CHAINS, CHAIN_MAXLEN, EMB_DIM
from atchley import encode_sequence
from encoders import ChainAutoencoder, ChainEncoder
from pose_vae import PoseVAERaw, vae_raw_loss
from pose_cvae import ConditionalPoseVAERaw, cvae_loss_masked
from data import PairData, add_mismatch
from geometry import quat_to_matrix, canonicalize_quat, geodesic_angle
import torch.nn.functional as F
import math, pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAD2DEG = 180.0/math.pi

# ---- pretraining ----
def pretrain_tcr_encoders(seq_df, epochs=12, bs=256, lr=1e-3, seed=42):
    torch.manual_seed(seed); states = {}
    for ch in CHAINS:
        X = np.stack([encode_sequence(s, CHAIN_MAXLEN[ch]) for s in seq_df[ch]]).astype("float32")
        loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs, shuffle=True)
        ae = ChainAutoencoder(CHAIN_MAXLEN[ch]).to(DEVICE); opt = torch.optim.Adam(ae.parameters(), lr=lr); ae.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(DEVICE); rec, _ = ae(xb); loss = F.mse_loss(rec, xb); opt.zero_grad(); loss.backward(); opt.step()
        states[ch] = {k: v.cpu() for k, v in ae.encoder.state_dict().items()}
    return states

def make_warm_start(tcr_enc, cpose_state=None, pep_state=None):
    """Returns a warm_start(model) closure that copies pretrained weights into a
    fresh model. Loads chain encoders (always), and -- when provided -- the
    peptide encoder and the conditional pose VAE (`cpose`), so a Cond* model can
    be warm-started end-to-end rather than trained from scratch. All loads are
    guarded by hasattr, so the same closure works for seq / pose / cond models."""
    def warm_start(model):
        if tcr_enc and hasattr(model, "encoders"):
            for c in model.encoders:
                if c in tcr_enc: model.encoders[c].load_state_dict(tcr_enc[c])
        if pep_state is not None and hasattr(model, "pep_encoder"):
            model.pep_encoder.load_state_dict(pep_state)
        if cpose_state is not None and hasattr(model, "cpose"):
            model.cpose.load_state_dict(cpose_state)
        return model
    return warm_start

def _pose_components(recon, x, mu, logvar):
    """Returns (rec_loss, kld, metrics) — rec matches vae_raw_loss; metrics in
    interpretable units: reach RMSE (scaled), direction & rotation error in degrees."""
    rec = x.new_zeros(()); reach_se = []; dir_d = []; rot_d = []
    for i, (rh, dh, Rh) in enumerate(recon):
        seg = x[:, i*7:(i+1)*7]; t, q = seg[:, 0:3], seg[:, 3:7]
        reach = torch.linalg.norm(t, dim=-1, keepdim=True); d = t/reach.clamp_min(1e-8); R = quat_to_matrix(canonicalize_quat(q))
        l_reach = F.mse_loss(rh, reach); cos = (dh*d).sum(-1).clamp(-1, 1); ga = geodesic_angle(Rh, R)
        rec = rec + l_reach + (1-cos).mean() + ga.mean()
        reach_se.append(l_reach.detach()); dir_d.append(torch.arccos(cos.clamp(-1+1e-6, 1-1e-6)).mean().detach()*RAD2DEG); rot_d.append(ga.mean().detach()*RAD2DEG)
    kld = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    m = {"reach_rmse": float(torch.stack(reach_se).mean().sqrt()),
         "dir_deg": float(torch.stack(dir_d).mean()), "rot_deg": float(torch.stack(rot_d).mean())}
    return rec, kld, m

@torch.no_grad()
def _latent_health(vae, X, bs=1024):
    """Posterior-collapse check: per-dim KL and #active latent dims (KL>0.01)."""
    vae.eval(); x = torch.tensor(X[:bs], dtype=torch.float32).to(DEVICE); _, mu, lv, _ = vae(x)
    kl_dim = 0.5*(mu.pow(2)+lv.exp()-1-lv).mean(0)
    return int((kl_dim > 0.01).sum().item()), float(kl_dim.mean().item())

def pretrain_posevae(pose_mat, rotation_encoder, epochs=40, bs=256, lr=1e-3, mask_p=0.3,
                     beta=0.1, seed=42, n_bodies=7, return_history=False, log=True,
                     early_stop=True, patience=8, min_delta=1e-3, monitor="recon"):
    """Masked-body pose VAE pretraining with plateau early stopping.
    Stops when `monitor` (default reconstruction loss) fails to improve by > min_delta
    for `patience` consecutive epochs; restores the best-epoch weights.
    With return_history=True, also returns a per-epoch DataFrame."""
    torch.manual_seed(seed); vae = PoseVAERaw(rotation_encoder=rotation_encoder).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(pose_mat)), batch_size=bs, shuffle=True)
    hist = []; best = float("inf"); wait = 0; best_state = None; best_ep = 0
    for ep in range(1, epochs+1):
        vae.train(); agg = defaultdict(float); ntot = 0
        for (x,) in loader:
            x = x.to(DEVICE); xin = x.clone()
            for bi in range(n_bodies):
                mm = torch.rand(x.size(0), device=DEVICE) < mask_p; xin[mm, bi*7:(bi+1)*7] = 0.0
            recon, mu, lv, _ = vae(xin); rec, kld, m = _pose_components(recon, x, mu, lv)
            loss = rec + beta*kld; opt.zero_grad(); loss.backward(); opt.step()
            nb = x.size(0); ntot += nb
            agg["loss"] += loss.item()*nb; agg["recon"] += rec.item()*nb; agg["kld"] += kld.item()*nb
            for k in ("reach_rmse", "dir_deg", "rot_deg"): agg[k] += m[k]*nb
        row = {k: agg[k]/ntot for k in agg}; row["epoch"] = ep
        row["active_units"], row["kl_per_dim"] = _latent_health(vae, pose_mat)
        hist.append(row)
        if log and (ep % 5 == 0 or ep == 1):
            print(f"  ep{ep:3d} loss={row['loss']:.3f} recon={row['recon']:.3f} KL={row['kld']:.3f} | "
                  f"rot={row['rot_deg']:.1f}deg dir={row['dir_deg']:.1f}deg reach_rmse={row['reach_rmse']:.3f} | "
                  f"active={row['active_units']}/{vae.latent_dim} kldim={row['kl_per_dim']:.3f}")
        # plateau early stopping on the monitored training loss
        cur = row[monitor]
        if cur < best - min_delta:
            best = cur; wait = 0; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
        elif early_stop:
            wait += 1
            if wait >= patience:
                if log: print(f"  early stop at epoch {ep} (no {monitor} improvement >{min_delta} for {patience} epochs; best epoch {best_ep})")
                break
    if best_state is not None:
        vae.load_state_dict(best_state)
    vae.eval()
    return (vae, pd.DataFrame(hist)) if return_history else vae

@torch.no_grad()
def _build_cond(enc_bank, pep_encoder, batch, chains, cond_on):
    """Reproduce across_peptide._cond from a fixed encoder bank (encoders in eval)."""
    p = pep_encoder(batch["pep"])
    if cond_on == "peptide":
        return p
    enc = {c: enc_bank[c](batch[c]) for c in ("va", "vb")}
    return torch.cat([enc["va"], enc["vb"], p], dim=1)   # "vpep"

def pretrain_cond_posevae(pool, trans_scale, rotation_encoder, cond_on="vpep",
                          warm_encoders=None, emb_dim=EMB_DIM, chains=CHAINS,
                          epochs=40, bs=256, lr=1e-3, mask_p=0.3, beta=0.1, seed=42,
                          n_bodies=7, posed_only=True, return_extras=False, log=True,
                          early_stop=True, patience=8, min_delta=1e-3, monitor="recon"):
    """Conditional twin of `pretrain_posevae`: warm-start initializer for the
    conditional pose VAE (`ConditionalPoseVAERaw`) used by the Cond* models.

    The conditioning `cond` is built exactly as the Cond* models build it
    (cond_on in {'peptide','vpep'}), from a FIXED encoder bank: chain encoders
    warm-started from `warm_encoders` (frozen here) and a seeded peptide encoder.
    The conditional VAE is then trained on masked-body reconstruction
    (`cvae_loss_masked`) with plateau early-stopping, mirroring the unconditioned
    routine. This is an *initializer only* -- downstream you warm-start it into a
    Cond* model and keep training end-to-end (cond drifts as the encoders move),
    so do NOT freeze it afterwards.

    Pass `warm_encoders` = the dict from `pretrain_tcr_encoders` for a meaningful
    conditioning signal. With `return_extras=True`, also returns the conditioning
    states ({'cpose', 'pep'}) so the model's `cpose` and `pep_encoder` can be
    warm-started consistently via `make_warm_start(tcr_enc, cpose_state, pep_state)`.

    `posed_only` keeps real-structure rows (pose_mask==1), matching the
    pose-discrimination convention (mismatch/null-pose negatives off)."""
    torch.manual_seed(seed)
    # fixed conditioning bank (not updated during pretraining)
    enc_bank = {c: ChainEncoder(emb_dim=emb_dim).to(DEVICE) for c in chains}
    pep_encoder = ChainEncoder(emb_dim=emb_dim).to(DEVICE)
    if warm_encoders:
        for c in enc_bank:
            if c in warm_encoders: enc_bank[c].load_state_dict(warm_encoders[c])
    for m in (*enc_bank.values(), pep_encoder):
        m.eval()
        for pr in m.parameters(): pr.requires_grad = False

    cdim = emb_dim if cond_on == "peptide" else 3*emb_dim
    cpose = ConditionalPoseVAERaw(cond_dim=cdim, rotation_encoder=rotation_encoder).to(DEVICE)
    opt = torch.optim.Adam(cpose.parameters(), lr=lr)

    frame = (pool[pool["pose_mask"] == 1] if posed_only else pool).reset_index(drop=True)
    loader = DataLoader(PairData(frame, trans_scale), batch_size=bs, shuffle=True)

    hist = []; best = float("inf"); wait = 0; best_state = None; best_ep = 0
    for ep in range(1, epochs+1):
        cpose.train(); agg = defaultdict(float); ntot = 0
        for b, _ in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            cond = _build_cond(enc_bank, pep_encoder, b, chains, cond_on)
            x = b["pose"]; mask = b["pose_mask"]; xin = x.clone()
            for bi in range(n_bodies):
                mm = torch.rand(x.size(0), device=DEVICE) < mask_p; xin[mm, bi*7:(bi+1)*7] = 0.0
            recon, mu, lv, _ = cpose(xin, cond)
            loss = cvae_loss_masked(recon, x, mu, lv, mask, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            # interpretable metrics over posed rows (all rows when posed_only)
            mk = mask.bool()
            if mk.any():
                rec, kld, met = _pose_components([(rh[mk], dh[mk], Rh[mk]) for (rh, dh, Rh) in recon],
                                                 x[mk], mu[mk], lv[mk])
            else:
                rec = kld = torch.zeros((), device=DEVICE); met = {"reach_rmse": 0., "dir_deg": 0., "rot_deg": 0.}
            nb = x.size(0); ntot += nb
            agg["loss"] += loss.item()*nb; agg["recon"] += float(rec)*nb; agg["kld"] += float(kld)*nb
            for k in ("reach_rmse", "dir_deg", "rot_deg"): agg[k] += met[k]*nb
        row = {k: agg[k]/ntot for k in agg}; row["epoch"] = ep
        hist.append(row)
        if log and (ep % 5 == 0 or ep == 1):
            print(f"  [cpose/{cond_on}] ep{ep:3d} loss={row['loss']:.3f} recon={row['recon']:.3f} "
                  f"KL={row['kld']:.3f} | rot={row['rot_deg']:.1f}deg dir={row['dir_deg']:.1f}deg "
                  f"reach_rmse={row['reach_rmse']:.3f}")
        cur = row[monitor]
        if cur < best - min_delta:
            best = cur; wait = 0; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in cpose.state_dict().items()}
        elif early_stop:
            wait += 1
            if wait >= patience:
                if log: print(f"  early stop at epoch {ep} (no {monitor} improvement >{min_delta} "
                              f"for {patience} epochs; best epoch {best_ep})")
                break
    if best_state is not None:
        cpose.load_state_dict(best_state)
    cpose.eval()
    if return_extras:
        extras = {"cpose": {k: v.detach().cpu().clone() for k, v in cpose.state_dict().items()},
                  "pep": {k: v.detach().cpu().clone() for k, v in pep_encoder.state_dict().items()},
                  "history": pd.DataFrame(hist)}
        return cpose, extras
    return cpose

# ---- supervised training ----
def predict(model, loader):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for b, y in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}; ps.append(torch.sigmoid(model(b)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def train_one(model, loader, epochs, lr, pw):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE)); model.train()
    for _ in range(epochs):
        for b, y in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}; y = y.to(DEVICE)
            opt.zero_grad(); crit(model(b), y).backward(); opt.step()
    return model

def train_one_aux(model, loader, epochs, lr, pw, lam_pose=0.3, beta=0.1):
    """classification + lambda*reconstruction + beta*KL (masked to structure-present rows)."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE)); model.train()
    for _ in range(epochs):
        for b, y in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}; y = y.to(DEVICE)
            logit, recon, mu, lv = model(b, return_aux=True)
            loss = crit(logit, y) + lam_pose*cvae_loss_masked(recon, b["pose"], mu, lv, b["pose_mask"], beta)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def run_repeated(factory, frame, splits, trans_scale, epochs=15, lr=1e-3, augment=False, aux=False):
    y = frame.label.to_numpy().astype(int); Y = []; P = []; IDX = []; aucs = []; prcs = []
    trainer = train_one_aux if aux else train_one
    for tr, te in splits:
        trf = add_mismatch(frame.iloc[tr]) if augment else frame.iloc[tr]
        model = factory().to(DEVICE)
        tl = DataLoader(PairData(trf.reset_index(drop=True), trans_scale), batch_size=64, shuffle=True)
        vl = DataLoader(PairData(frame.iloc[te].reset_index(drop=True), trans_scale), batch_size=128)
        yt = trf.label.to_numpy().astype(int); pw = (yt == 0).sum()/max((yt == 1).sum(), 1)
        trainer(model, tl, epochs, lr, pw); pr, _ = predict(model, vl); yte = y[te]
        Y.append(yte); P.append(pr); IDX.append(np.asarray(te))
        if len(np.unique(yte)) > 1: aucs.append(roc_auc_score(yte, pr)); prcs.append(average_precision_score(yte, pr))
    return np.concatenate(Y), np.concatenate(P), np.concatenate(IDX), np.array(aucs), np.array(prcs)

# ---- honest bootstrap ----
def aggregate_oof(idx, p, n):
    psum = np.zeros(n); pcnt = np.zeros(n); np.add.at(psum, idx, p); np.add.at(pcnt, idx, 1)
    tested = pcnt > 0; pagg = np.full(n, np.nan); pagg[tested] = psum[tested]/pcnt[tested]; return pagg, tested

def cluster_bootstrap(y, pa, pb, groups, tested, n=2000, seed=42):
    rng = np.random.default_rng(seed); g2i = defaultdict(list)
    for i in np.where(tested)[0]: g2i[int(groups[i])].append(i)
    gids = list(g2i); dR = []; dP = []
    for _ in range(n):
        samp = np.concatenate([g2i[g] for g in rng.choice(gids, len(gids), True)]); ys = y[samp]
        if len(np.unique(ys)) < 2: continue
        dR.append(roc_auc_score(ys, pa[samp])-roc_auc_score(ys, pb[samp]))
        dP.append(average_precision_score(ys, pa[samp])-average_precision_score(ys, pb[samp]))
    def s(d):
        d = np.array(d)
        if len(d) == 0: return (float("nan"),)*4
        return (round(float(d.mean()), 4), round(float(np.percentile(d, 2.5)), 4),
                round(float(np.percentile(d, 97.5)), 4), round(float(2*min((d <= 0).mean(), (d >= 0).mean())), 4))
    return s(dR), s(dP)
