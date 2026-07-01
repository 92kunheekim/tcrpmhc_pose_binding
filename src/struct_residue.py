"""StructResidueFusion: fuse CDR3 sequence + structure-augmented peptide/CDR3 residue
encoders + the (frozen, pre-fit) conditional-flow V-marginalized CDR3 body residual `z`.

Ablation ladder (blocks):
  seq              : pep_seq + cdr3_seq            (no structure baseline)
  seq_pepstruct    : pep_struct + cdr3_seq         (structural peptide instead of Atchley)
  seq_pepstruct_z  : + z (body-level V-marginalized flow residual)
  full             : + cdr3_struct (per-residue CDR3 geometry, pLDDT-gated)

Design decisions baked in: the conditional flow is PRE-FIT and FROZEN (label-agnostic
p(pose|V)); the body-level residual is V-marginalized; the per-residue CDR3 channel is
NOT V-marginalized (deliberate); relaxed-structure residues, pLDDT-gated.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CHAIN_MAXLEN, PEP_MAXLEN, EMB_DIM
from atchley import encode_sequence
from encoders import ChainEncoder
from residue_encoder import residue_arrays, ResidueSetEncoder
from pan_specific import (PanFusion, pan_loader, cache_tower_features,
                          fit_cond_flow, flow_residual, DEVICE)

CDR3_LEN = CHAIN_MAXLEN["cdr3a"]                      # 25

STRUCT_BLOCKS = {
    "seq":             ("pep_seq", "cdr3_seq"),
    "seq_pepstruct":   ("pep_struct", "cdr3_seq"),
    "seq_pepstruct_z": ("pep_struct", "cdr3_seq", "z"),
    "full":            ("pep_struct", "cdr3_seq", "cdr3_struct", "z"),
}


class StructResidueFusion(nn.Module):
    _B = ("pep_seq", "pep_struct", "cdr3_seq", "cdr3_struct", "z")

    def __init__(self, emb_dim=EMB_DIM, z_dim=12, hidden=128, dropout=0.3,
                 pep_len=PEP_MAXLEN, cdr3_len=CDR3_LEN, blocks="full"):
        super().__init__()
        self.blocks = STRUCT_BLOCKS[blocks] if isinstance(blocks, str) else tuple(blocks)
        self.pep_seq_enc = ChainEncoder(emb_dim=emb_dim)
        self.cdr3a_seq_enc = ChainEncoder(emb_dim=emb_dim)
        self.cdr3b_seq_enc = ChainEncoder(emb_dim=emb_dim)
        self.pep_struct_enc = ResidueSetEncoder(emb_dim=emb_dim, max_len=pep_len)
        self.cdr3_struct_enc = ResidueSetEncoder(emb_dim=emb_dim, max_len=cdr3_len)   # shared for a/b
        self.z_mlp = nn.Sequential(nn.Linear(z_dim, emb_dim), nn.ReLU())
        dims = {"pep_seq": emb_dim, "pep_struct": emb_dim, "cdr3_seq": 2*emb_dim,
                "cdr3_struct": 2*emb_dim, "z": emb_dim}
        self.head = nn.Sequential(nn.Linear(sum(dims[b] for b in self.blocks), hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, b):
        f = {}
        if "pep_seq" in self.blocks:
            f["pep_seq"] = self.pep_seq_enc(b["pep_seq"])
        if "pep_struct" in self.blocks:
            f["pep_struct"] = self.pep_struct_enc(b["pep_res"], b["pep_mask"], b["pep_conf"])
        if "cdr3_seq" in self.blocks:
            f["cdr3_seq"] = torch.cat([self.cdr3a_seq_enc(b["cdr3a_seq"]),
                                       self.cdr3b_seq_enc(b["cdr3b_seq"])], 1)
        if "cdr3_struct" in self.blocks:
            f["cdr3_struct"] = torch.cat([self.cdr3_struct_enc(b["a_res"], b["a_mask"], b["a_conf"]),
                                          self.cdr3_struct_enc(b["b_res"], b["b_mask"], b["b_conf"])], 1)
        if "z" in self.blocks:
            f["z"] = self.z_mlp(b["z"])
        return self.head(torch.cat([f[k] for k in self.blocks], 1)).squeeze(-1)


def warm_cdr3(model, tcr_enc):
    """Load pretrained cdr3a/cdr3b chain encoders into the sequence towers (optional)."""
    if tcr_enc and "cdr3a" in tcr_enc: model.cdr3a_seq_enc.load_state_dict(tcr_enc["cdr3a"])
    if tcr_enc and "cdr3b" in tcr_enc: model.cdr3b_seq_enc.load_state_dict(tcr_enc["cdr3b"])
    return model


class _DictDS(Dataset):
    def __init__(self, d): self.d = d; self.n = len(d["y"])
    def __len__(self): return self.n
    def __getitem__(self, i): return {k: v[i] for k, v in self.d.items()}


def build_struct_tensors(frame, res_df, z, cdr3_len=CDR3_LEN):
    """Assemble the per-example tensor dict for StructResidueFusion, aligned to frame order.
    frame needs columns: pep, cdr3a, cdr3b, label, id. `z` is (N, z_dim) flow residual."""
    ids = frame["id"].tolist()
    def enc(col, L): return np.stack([encode_sequence(s, L) for s in frame[col]]).astype("float32")
    Xp, Mp, Cp = residue_arrays(res_df, "peptide", ids, PEP_MAXLEN)
    Xa, Ma, Ca = residue_arrays(res_df, "cdr3a", ids, cdr3_len)
    Xb, Mb, Cb = residue_arrays(res_df, "cdr3b", ids, cdr3_len)
    t = lambda a: torch.from_numpy(np.asarray(a, "float32"))
    return {
        "pep_seq": t(enc("pep", PEP_MAXLEN)),
        "cdr3a_seq": t(enc("cdr3a", CDR3_LEN)), "cdr3b_seq": t(enc("cdr3b", CDR3_LEN)),
        "pep_res": t(Xp), "pep_mask": t(Mp), "pep_conf": t(Cp),
        "a_res": t(Xa), "a_mask": t(Ma), "a_conf": t(Ca),
        "b_res": t(Xb), "b_mask": t(Mb), "b_conf": t(Cb),
        "z": t(np.asarray(z, "float32")),
        "y": t(frame["label"].to_numpy("float32")),
    }


def _train(model, dd, epochs, lr, pw, bs, wd=1e-4):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))
    dl = DataLoader(_DictDS(dd), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for b in dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            loss = crit(model(b), b["y"])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def _predict(model, dd, bs=1024):
    model.to(DEVICE).eval(); ps, ys = [], []
    for b in DataLoader(_DictDS(dd), batch_size=bs, shuffle=False):
        b = {k: v.to(DEVICE) for k, v in b.items()}
        ps.append(torch.sigmoid(model(b)).cpu().numpy()); ys.append(b["y"].cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)


def permute_within_peptide(z, peptides, seed=0):
    """Shuffle rows of z within each peptide group (pose-permutation control)."""
    rng = np.random.default_rng(seed); z = np.array(z); out = z.copy()
    for p in np.unique(peptides):
        idx = np.where(peptides == p)[0]; out[idx] = z[rng.permutation(idx)]
    return out


def run_struct_ablation(pool, res_df, trans_scale, esm_table, splits, warm=None, tcr_enc=None,
                        blocks=("seq", "seq_pepstruct", "seq_pepstruct_z", "full"),
                        epochs=20, lr=1e-3, bs=128, flow_epochs=150, permute_z=False,
                        pep_arch="transformer", mhc_col="mhc_seq", log=True, seed=0):
    """Per fold: cache towers -> PRE-FIT + FREEZE conditional flow on TRAIN -> z (train/test)
    -> train each `blocks` config of StructResidueFusion -> collect OOF. permute_z adds a
    pose-permutation control (z shuffled within peptide). Returns (metrics_df, oof)."""
    import pandas as pd
    def fresh_pan():
        m = PanFusion(use_pose=True, pep_arch=pep_arch)
        return warm(m) if warm is not None else m

    configs = list(blocks) + (["full_permz"] if permute_z else [])
    rows = []; oof = {c: {"idx": [], "y": [], "p": []} for c in configs}
    for si, (tr, te) in enumerate(splits):
        trf = pool.iloc[tr].reset_index(drop=True); tef = pool.iloc[te].reset_index(drop=True)
        yt = trf.label.to_numpy().astype(int); pw = (yt == 0).sum()/max((yt == 1).sum(), 1)
        base = fresh_pan()
        ctr = cache_tower_features(base, pan_loader(trf, trans_scale, esm_table, shuffle=False, mhc_col=mhc_col))
        cte = cache_tower_features(base, pan_loader(tef, trans_scale, esm_table, shuffle=False, mhc_col=mhc_col))
        fit = fit_cond_flow(ctr, epochs=flow_epochs, log=False)      # PRE-FIT + frozen
        z_tr, _ = flow_residual(fit, ctr); z_te, _ = flow_residual(fit, cte)
        dtr = build_struct_tensors(trf, res_df, z_tr); dte = build_struct_tensors(tef, res_df, z_te)
        for cfg in configs:
            blk = "full" if cfg == "full_permz" else cfg
            dtr_c, dte_c = dtr, dte
            if cfg == "full_permz":                                  # control: shuffle z within peptide
                dtr_c = {**dtr, "z": torch.from_numpy(permute_within_peptide(z_tr, trf.peptide.to_numpy(), seed))}
                dte_c = {**dte, "z": torch.from_numpy(permute_within_peptide(z_te, tef.peptide.to_numpy(), seed))}
            m = StructResidueFusion(blocks=blk, z_dim=z_tr.shape[1])
            if tcr_enc is not None: warm_cdr3(m, tcr_enc)
            _train(m, dtr_c, epochs, lr, pw, bs)
            p, y = _predict(m, dte_c)
            oof[cfg]["idx"].append(np.asarray(te)); oof[cfg]["y"].append(y); oof[cfg]["p"].append(p)
            au = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            rows.append(dict(config=cfg, split=si, auroc=au, auprc=ap, n=len(y), prevalence=float(y.mean())))
            if log: print(f"  split {si} [{cfg:16s}] AUROC={au:.3f} AUPRC={ap:.3f}")
    for c in configs:
        oof[c] = {k: np.concatenate(v) for k, v in oof[c].items()}
    return pd.DataFrame(rows), oof
