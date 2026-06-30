"""Training, pretraining, repeated leakage-controlled CV, and cluster bootstrap."""
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CHAINS, CHAIN_MAXLEN
from atchley import encode_sequence
from encoders import ChainAutoencoder
from pose_vae import PoseVAERaw, vae_raw_loss
from pose_cvae import cvae_loss_masked
from data import PairData, add_mismatch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def make_warm_start(tcr_enc):
    def warm_start(model):
        if tcr_enc:
            for c in model.encoders:
                if c in tcr_enc: model.encoders[c].load_state_dict(tcr_enc[c])
        return model
    return warm_start

def pretrain_posevae(pose_mat, rotation_encoder, epochs=25, bs=256, lr=1e-3, mask_p=0.3, seed=42, n_bodies=7):
    torch.manual_seed(seed); vae = PoseVAERaw(rotation_encoder=rotation_encoder).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(pose_mat)), batch_size=bs, shuffle=True); vae.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(DEVICE); xin = x.clone()
            for bi in range(n_bodies):
                mm = torch.rand(x.size(0), device=DEVICE) < mask_p; xin[mm, bi*7:(bi+1)*7] = 0.0
            recon, mu, lv, _ = vae(xin); loss = vae_raw_loss(recon, x, mu, lv); opt.zero_grad(); loss.backward(); opt.step()
    vae.eval(); return vae

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
