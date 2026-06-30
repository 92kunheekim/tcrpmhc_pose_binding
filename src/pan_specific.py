"""PAN-allele model (class I only) — across peptide AND HLA allele.

Adds an MHC tower so the model can generalize over class-I HLA alleles, not just
A*02:01. The MHC encoder uses **ESM-2** on the **first 150 AA of the heavy chain**
(the alpha1+alpha2 groove region — the polymorphic, peptide/TCR-contacting domain,
matching the FramePose groove frame residues 1-150). ESM-2 is frozen and its
embeddings are **precomputed offline per unique heavy-chain sequence** (there are few
class-I alleles), then a small MLP projects them to emb_dim — mirroring pMTnet-omni's
use of ESM-2 for pan-MHC generalization.

The peptide tower is FiLM-conditioned on the MHC embedding (same peptide is presented
differently by different alleles). The pose is relative to the pMHC groove, so it is
conditioned on [Va, Vb, peptide, MHC]. Fusion head sees [t, peptide|MHC, mhc,
t*(peptide|MHC), pose_latent].

Requires the data to carry the class-I heavy-chain sequence (column `mhc_seq` in
combined_sorted.csv). NOTE: the current A*02:01-only cohort has a constant MHC, so
the MHC tower only becomes informative once multi-allele data is added; this module
makes the architecture ready.

Evaluation for a pan model: leave-one-allele-out (LOAO) + leave-one-peptide-out.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from config import CHAINS, CHAIN_MAXLEN, PEP_MAXLEN, EMB_DIM
from atchley import encode_sequence, ATCHLEY
from encoders import ChainEncoder
from pose_cvae import ConditionalPoseVAERaw, cvae_loss_masked
from data import build_pose

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA_ORDER = list(ATCHLEY.keys())                       # 20 AAs, fixed index order
_AA2I = {a: i for i, a in enumerate(AA_ORDER)}

MHC_MAXLEN = 150                          # first 150 AA of the class-I heavy chain (alpha1+alpha2 groove)
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"  # 1280-d; drop to t30_150M (640) / t12_35M (480) for speed
ESM_DIM = 1280


def compute_esm_embeddings(seqs, model_name=ESM_MODEL, device=None, maxlen=MHC_MAXLEN, batch=8):
    """Frozen ESM-2 mean-pooled embedding of the first `maxlen` AA of each unique
    heavy-chain sequence. Returns {full_seq: np.ndarray(esm_dim)}. Run ONCE offline;
    cache to disk and reuse. Requires `pip install transformers`."""
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    uniq = sorted(set(seqs)); emb = {}
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            block = uniq[i:i+batch]
            t = tok([s[:maxlen] for s in block], return_tensors="pt", padding=True,
                    truncation=True, max_length=maxlen+2).to(device)
            out = mdl(**t).last_hidden_state            # (B, L, D)
            m = t.attention_mask.unsqueeze(-1).float()
            pooled = (out*m).sum(1)/m.sum(1).clamp_min(1.0)   # mean over real tokens
            for s, v in zip(block, pooled.cpu().numpy()): emb[s] = v
    return emb


class MHCEncoder(nn.Module):
    """Class-I MHC tower: projects a precomputed frozen ESM-2 embedding (first 150 AA
    of the heavy chain) to emb_dim. ESM-2 itself is not trained (embeddings are cached)."""
    def __init__(self, emb_dim=EMB_DIM, esm_dim=ESM_DIM, hidden=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(esm_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, emb_dim))
    def forward(self, mhc_esm):              # (B, esm_dim) precomputed
        return self.proj(mhc_esm)


class MHCConditionedPeptideEncoder(nn.Module):
    """Peptide tower FiLM-conditioned on the MHC embedding: the same peptide is
    presented differently by different alleles, so the peptide embedding is modulated
    by the MHC. peptide_emb = (1 + gamma(mhc)) * ChainEncoder(peptide) + beta(mhc).
    FiLM layer is zero-initialized -> starts as the plain (unconditioned) peptide encoder."""
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.pep = ChainEncoder(emb_dim=emb_dim)
        self.film = nn.Linear(emb_dim, 2*emb_dim)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
    def forward(self, pep_atchley, mhc_emb):
        z = self.pep(pep_atchley)
        gamma, beta = self.film(mhc_emb).chunk(2, dim=-1)
        return (1.0 + gamma)*z + beta


class PeptideEncoderTransformer(nn.Module):
    """Per-position peptide transformer (stronger than the pooled ChainEncoder for masked
    modelling). Residue Atchley -> linear + learned position embedding, masked positions
    replaced by a learned [MASK] token, FiLM-conditioned on the MHC embedding (per position),
    then a small TransformerEncoder over positions.
      forward(...) -> pooled emb (mean over real residues -> Linear): drop-in peptide tower.
      forward(..., return_tokens=True) -> per-position states (B,L,emb): for the MLM head.
    Padding (all-zero Atchley rows) is masked from attention; the FiLM layer is zero-init
    so it starts as a plain (MHC-agnostic) transformer."""
    def __init__(self, emb_dim=EMB_DIM, n_layers=2, n_heads=4, ff=128, max_len=PEP_MAXLEN, dropout=0.1):
        super().__init__(); self.d = emb_dim; self.max_len = max_len
        self.res_proj = nn.Linear(5, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, emb_dim)); nn.init.normal_(self.pos_emb, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(emb_dim)); nn.init.normal_(self.mask_token, std=0.02)
        self.film = nn.Linear(emb_dim, 2*emb_dim); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        layer = nn.TransformerEncoderLayer(emb_dim, n_heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
    def _encode(self, pep, mhc_emb, mask=None):
        pad = (pep.abs().sum(-1) == 0)                            # (B,L) True = padding
        x = self.res_proj(pep) + self.pos_emb[:pep.size(1)].unsqueeze(0)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), x)
        g, b = self.film(mhc_emb).chunk(2, dim=-1)                # per-position FiLM (broadcast)
        x = (1.0 + g).unsqueeze(1)*x + b.unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=pad), pad
    def forward(self, pep_atchley, mhc_emb, mask=None, return_tokens=False):
        tok, pad = self._encode(pep_atchley, mhc_emb, mask)
        if return_tokens:
            return tok
        keep = (~pad).float().unsqueeze(-1)
        pooled = (tok*keep).sum(1) / keep.sum(1).clamp_min(1.0)   # mean over real residues
        return self.out_proj(pooled)


# Head feature blocks (each emb_dim, except `pose` = cpose.latent_dim):
#   t   = TCR embedding         p = peptide (MHC-conditioned)   h = MHC
#   tp  = t * p (TCR x peptide) th = t * h (TCR x MHC)          pose = conditioned pose_mu
# Presets for the ablation requested (isolate pose vs sequence contribution):
HEAD_PRESETS = {
    "pose":       ("pose",),                              # pose only
    "pose_tp_th": ("pose", "tp", "th"),                  # pose + interaction terms
    "pose_tp":    ("pose", "tp"),                        # pose + TCRxpeptide
    "full":       ("t", "p", "h", "tp", "th", "pose"),   # everything
}


class PanFusion(nn.Module):
    """seq + peptide + MHC (+ optional [V,peptide,MHC]-conditioned pose), class I.

    use_pose=False -> seq+pep+mhc baseline (no pose tower).
    rotation_encoder in {'6d','qnn'}; pose conditioned on cond=[Va,Vb,peptide,MHC].
    head_feats selects which blocks feed the classifier head -- a name from HEAD_PRESETS
    or any tuple drawn from {t,p,h,tp,th,pose}. Towers are always computed (needed for the
    interaction terms and pose conditioning); only the head input changes. `pose` entries
    are dropped automatically when use_pose=False.
    NOTE: under a single allele h is constant, so th = t*h is ~collinear with t -- include
    it only with multi-allele data.
    """
    _FEATS = ("t", "p", "h", "tp", "th", "pose")

    def __init__(self, emb_dim=EMB_DIM, hidden_head=128, dropout=0.3, chains=CHAINS,
                 use_pose=True, rotation_encoder="6d", esm_dim=ESM_DIM,
                 head_feats="full", pep_arch="transformer"):
        super().__init__(); self.chains = chains; self.use_pose = use_pose
        self.encoders = nn.ModuleDict({c: ChainEncoder(emb_dim=emb_dim) for c in chains})
        self.mhc_encoder = MHCEncoder(emb_dim=emb_dim, esm_dim=esm_dim)   # projects precomputed ESM-2 vec
        # peptide tower (MHC-conditioned). 'transformer' = per-position MLM-pretrainable; 'conv' = pooled ChainEncoder
        self.pep_encoder = (PeptideEncoderTransformer(emb_dim=emb_dim) if pep_arch == "transformer"
                            else MHCConditionedPeptideEncoder(emb_dim=emb_dim))
        self.tcr_proj = nn.Linear(emb_dim*len(chains), emb_dim)
        pose_dim = 0
        if use_pose:
            self.cpose = ConditionalPoseVAERaw(cond_dim=4*emb_dim, rotation_encoder=rotation_encoder)  # [Va,Vb,pep,mhc]
            pose_dim = self.cpose.latent_dim; self.null_pose = nn.Parameter(torch.zeros(pose_dim))
        # resolve head feature selection
        feats = HEAD_PRESETS[head_feats] if isinstance(head_feats, str) else tuple(head_feats)
        bad = [f for f in feats if f not in self._FEATS]
        if bad: raise ValueError(f"unknown head_feats {bad}; choose from {self._FEATS}")
        feats = tuple(f for f in feats if f != "pose" or use_pose)   # drop pose if no pose tower
        if not feats: raise ValueError("head_feats is empty (pose requires use_pose=True)")
        self.head_feats = feats
        dims = {"t": emb_dim, "p": emb_dim, "h": emb_dim, "tp": emb_dim, "th": emb_dim, "pose": pose_dim}
        head_in = sum(dims[f] for f in feats)
        self.head = nn.Sequential(nn.Linear(head_in, hidden_head), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_head, 1))

    def forward(self, b, return_aux=False):
        enc = {c: self.encoders[c](b[c]) for c in self.chains}
        t = self.tcr_proj(torch.cat([enc[c] for c in self.chains], 1))
        h = self.mhc_encoder(b["mhc"])              # MHC first
        p = self.pep_encoder(b["pep"], h)           # peptide conditioned on MHC
        recon = mu = lv = None; pose = None
        if self.use_pose:
            cond = torch.cat([enc["va"], enc["vb"], p, h], dim=1)
            recon, mu, lv, _ = self.cpose(b["pose"], cond); mask = b.get("pose_mask"); muf = mu
            if mask is not None:
                mk = mask.float().unsqueeze(1); muf = mk*mu + (1-mk)*self.null_pose.unsqueeze(0)
            pose = muf
        fmap = {"t": t, "p": p, "h": h, "tp": t*p, "th": t*h, "pose": pose}
        logit = self.head(torch.cat([fmap[f] for f in self.head_feats], 1)).squeeze(-1)
        return (logit, recon, mu, lv) if (return_aux and self.use_pose) else logit


class PanPairData(Dataset):
    """Batch with TCR chains, peptide, class-I MHC (precomputed ESM-2 vec), raw pose, mask, label.
    `esm_table` = {heavy_chain_seq: esm_vector} from compute_esm_embeddings."""
    def __init__(self, frame, trans_scale, esm_table, mhc_col="mhc_seq"):
        self.enc = {c: np.stack([encode_sequence(s, CHAIN_MAXLEN[c]) for s in frame[c]]).astype("float32") for c in CHAINS}
        self.pep = np.stack([encode_sequence(s, PEP_MAXLEN) for s in frame["pep"]]).astype("float32")
        self.mhc = np.stack([esm_table[s] for s in frame[mhc_col]]).astype("float32")   # precomputed ESM-2 (first 150 AA)
        self.pose = build_pose(frame, trans_scale); self.mask = frame["pose_mask"].to_numpy("float32")
        self.y = frame["label"].to_numpy("float32")
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        it = {c: torch.from_numpy(self.enc[c][i]) for c in CHAINS}
        it["pep"] = torch.from_numpy(self.pep[i]); it["mhc"] = torch.from_numpy(self.mhc[i])
        it["pose"] = torch.from_numpy(self.pose[i]); it["pose_mask"] = torch.tensor(self.mask[i])
        return it, torch.tensor(self.y[i])


# ----------------------------------------------------------------------------
# Self-supervised MHC-conditioned masked-residue peptide pretraining
# (learns allele-specific anchor motifs -> warm-starts pep_encoder + mhc_encoder)
# ----------------------------------------------------------------------------
class MHCMaskedPeptideModel(nn.Module):
    """BERT-style masked-residue model for peptides, conditioned on the class-I MHC.

    Pretraining task (self-supervised): randomly mask a fraction of peptide residues,
    then predict their amino-acid identity from the unmasked residues + the MHC.
    Because the allele constrains the *anchor* positions (e.g. P2=L/M, POmega=V/L for
    A*02:01), the FiLM conditioning is forced to encode allele-specific presentation
    motifs -- exactly the signal that a plain peptide-only reconstruction objective
    cannot learn. Reuses `MHCConditionedPeptideEncoder` and `MHCEncoder` so the trained
    weights warm-start the pan model's peptide and MHC towers directly.
    """
    def __init__(self, emb_dim=EMB_DIM, esm_dim=ESM_DIM, max_len=PEP_MAXLEN, n_aa=len(AA_ORDER)):
        super().__init__(); self.max_len = max_len; self.n_aa = n_aa
        self.mhc_encoder = MHCEncoder(emb_dim=emb_dim, esm_dim=esm_dim)
        self.pep_encoder = MHCConditionedPeptideEncoder(emb_dim=emb_dim)   # .pep + .film
        self.decoder = nn.Linear(emb_dim, max_len*n_aa)                    # per-position AA logits

    def forward(self, masked_pep, mhc_esm, mask=None):
        h = self.mhc_encoder(mhc_esm)                  # (B, emb)
        z = self.pep_encoder(masked_pep, h)            # MHC-conditioned peptide emb (B, emb)
        return self.decoder(z).view(-1, self.max_len, self.n_aa)   # (B, L, 20)


class MHCMaskedPeptideTransformer(nn.Module):
    """Stronger MLM: per-position TransformerEncoder (no pooling bottleneck) over the
    peptide, FiLM-conditioned on the MHC. Predicts each masked residue from its *context*
    + allele -> learns anchor motifs far better than the pooled model. Its `pep_encoder`
    (a PeptideEncoderTransformer) and `mhc_encoder` warm-start PanFusion (pep_arch='transformer')."""
    def __init__(self, emb_dim=EMB_DIM, esm_dim=ESM_DIM, max_len=PEP_MAXLEN, n_aa=len(AA_ORDER),
                 n_layers=2, n_heads=4, ff=128, dropout=0.1):
        super().__init__(); self.max_len = max_len; self.n_aa = n_aa
        self.mhc_encoder = MHCEncoder(emb_dim=emb_dim, esm_dim=esm_dim)
        self.pep_encoder = PeptideEncoderTransformer(emb_dim=emb_dim, n_layers=n_layers,
                                                     n_heads=n_heads, ff=ff, max_len=max_len, dropout=dropout)
        self.head = nn.Linear(emb_dim, n_aa)            # per-position AA logits

    def forward(self, pep, mhc_esm, mask=None):
        h = self.mhc_encoder(mhc_esm)
        tok = self.pep_encoder(pep, h, mask=mask, return_tokens=True)   # (B, L, emb)
        return self.head(tok)                                            # (B, L, 20)


def peptide_pretrain_arrays(df, esm_table, pep_col="peptide", mhc_col="mhc_seq"):
    """Build (pep_seqs, mhc_esm) for `pretrain_peptide_masked` from an eluted-ligand /
    presentation table. `df` needs a peptide column and a heavy-chain MHC-sequence column;
    `esm_table` = {heavy_chain_seq: esm_vector} from compute_esm_embeddings. Only the
    positive (presented) (peptide, allele) pairs are needed -- no binding labels/decoys."""
    pep_seqs = df[pep_col].astype(str).tolist()
    mhc_esm = np.stack([esm_table[s] for s in df[mhc_col]]).astype("float32")
    return pep_seqs, mhc_esm


def pretrain_peptide_masked(pep_seqs, mhc_esm, max_len=PEP_MAXLEN, mask_p=0.15,
                            epochs=40, bs=256, lr=1e-3, seed=42, device=None,
                            log=True, return_history=False, return_model=False,
                            early_stop=True, patience=8, min_delta=1e-4, arch="transformer"):
    """Self-supervised MHC-conditioned masked-residue pretraining of the peptide tower.

    pep_seqs : list[str] of presented peptides.
    mhc_esm  : (N, esm_dim) precomputed ESM-2 vectors of the presenting allele, aligned
               row-for-row with pep_seqs (use peptide_pretrain_arrays to build both).
    arch     : 'transformer' (per-position TransformerEncoder; stronger MLM, recommended)
               or 'pooled' (legacy ChainEncoder bottleneck).

    Dynamic BERT-style masking each step (mask_p of non-pad residues); cross-entropy on the
    masked positions only. Tracks masked-residue top-1 accuracy and plateau early-stops on
    training loss with best-weight restore.

    Returns (pep_state, mhc_state[, history_df][, model]) -- state dicts warm-start
    `model.pep_encoder`/`model.mhc_encoder` of PanFusion (pep_arch must match `arch`).
    Pass return_model=True to also get the trained MLM (e.g. for masked_accuracy_breakdown)."""
    import pandas as pd
    device = device or DEVICE
    torch.manual_seed(seed); np.random.seed(seed)

    # clean Atchley matrices + true AA indices (pad -> -100 so CE ignores them)
    X = np.stack([encode_sequence(s, max_len) for s in pep_seqs]).astype("float32")
    TI = np.full((len(pep_seqs), max_len), -100, dtype=np.int64)
    for n, s in enumerate(pep_seqs):
        for j, a in enumerate(str(s).strip().upper()[:max_len]):
            if a in _AA2I: TI[n, j] = _AA2I[a]
    M = np.asarray(mhc_esm, dtype="float32")
    ds = TensorDataset(torch.tensor(X), torch.tensor(TI), torch.tensor(M))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    is_tf = arch == "transformer"
    model = (MHCMaskedPeptideTransformer if is_tf else MHCMaskedPeptideModel)(
        esm_dim=M.shape[1], max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    hist, best, wait, best_state, best_ep = [], float("inf"), 0, None, 0
    for ep in range(1, epochs+1):
        model.train(); tot = corr = msk = n_tot = 0
        for xb, ti, mb in loader:
            xb, ti, mb = xb.to(device), ti.to(device), mb.to(device)
            nonpad = ti != -100
            mask = nonpad & (torch.rand(ti.shape, device=device) < mask_p)
            if not bool(mask.any()):                         # guarantee >=1 masked token
                idx = nonpad.float().argmax(1); mask[torch.arange(ti.size(0)), idx] = nonpad.any(1)
            if is_tf:                                        # transformer: pass mask -> [MASK] token
                logits = model(xb, mb, mask=mask)
            else:                                            # pooled: blank masked Atchley rows
                xin = xb.clone(); xin[mask] = 0.0; logits = model(xin, mb)
            target = torch.where(mask, ti, torch.full_like(ti, -100))
            loss = crit(logits.reshape(-1, model.n_aa), target.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = logits.argmax(-1)
                corr += int((pred[mask] == ti[mask]).sum()); msk += int(mask.sum())
            tot += float(loss) * xb.size(0); n_tot += xb.size(0)
        ep_loss = tot/max(n_tot, 1); ep_acc = corr/max(msk, 1)
        hist.append(dict(epoch=ep, loss=ep_loss, masked_acc=ep_acc))
        if log: print(f"[pep-mask:{arch}] ep {ep:3d}  loss {ep_loss:.4f}  masked_acc {ep_acc:.3f}")
        if ep_loss < best - min_delta:
            best, wait, best_ep = ep_loss, 0, ep
            best_state = {"pep": {k: v.detach().cpu().clone() for k, v in model.pep_encoder.state_dict().items()},
                          "mhc": {k: v.detach().cpu().clone() for k, v in model.mhc_encoder.state_dict().items()}}
        else:
            wait += 1
            if early_stop and wait >= patience:
                if log: print(f"[pep-mask:{arch}] early stop at ep {ep} (best ep {best_ep}, loss {best:.4f})")
                break
    if best_state is None:   # epochs==0 guard
        best_state = {"pep": model.pep_encoder.state_dict(), "mhc": model.mhc_encoder.state_dict()}
    model.pep_encoder.load_state_dict(best_state["pep"]); model.mhc_encoder.load_state_dict(best_state["mhc"])
    out = [best_state["pep"], best_state["mhc"]]
    if return_history: out.append(pd.DataFrame(hist))
    if return_model: out.append(model)
    return tuple(out) if (return_history or return_model) else (out[0], out[1])


@torch.no_grad()
def masked_accuracy_breakdown(model, pep_seqs, mhc_esm, alleles=None, device=None,
                              bs=512, max_len=PEP_MAXLEN):
    """Per-position and per-allele masked-residue top-1 accuracy. For each position, mask
    exactly that position (where a residue is present) and predict it from context + MHC.
    A working MLM shows accuracy *spikes at the anchor positions* (P2, P-Omega) well above the
    variable middle positions, even if the overall average is modest.

    model   : a trained MLM (MHCMaskedPeptideTransformer / MHCMaskedPeptideModel).
    alleles : optional array aligned with pep_seqs for the per-allele breakdown.
    Returns (per_position_df, per_allele_df_or_None)."""
    import pandas as pd
    from collections import defaultdict
    device = device or DEVICE; model.to(device).eval()
    N = len(pep_seqs)
    X = np.stack([encode_sequence(s, max_len) for s in pep_seqs]).astype("float32")
    TI = np.full((N, max_len), -100, dtype=np.int64)
    for n, s in enumerate(pep_seqs):
        for j, a in enumerate(str(s).strip().upper()[:max_len]):
            if a in _AA2I: TI[n, j] = _AA2I[a]
    Xt, TIt, Mt = torch.tensor(X), torch.tensor(TI), torch.tensor(np.asarray(mhc_esm, "float32"))
    al = np.asarray(alleles) if alleles is not None else None
    pos_corr = np.zeros(max_len); pos_tot = np.zeros(max_len)
    al_corr, al_tot = defaultdict(int), defaultdict(int)
    for pos in range(max_len):
        if int((TIt[:, pos] != -100).sum()) == 0: continue
        for i in range(0, N, bs):
            xb = Xt[i:i+bs].to(device); ti = TIt[i:i+bs].to(device); mb = Mt[i:i+bs].to(device)
            v = ti[:, pos] != -100
            if not bool(v.any()): continue
            mask = torch.zeros(xb.shape[:2], dtype=torch.bool, device=device); mask[:, pos] = v
            pred = model(xb, mb, mask=mask)[:, pos].argmax(-1)
            ok = (pred == ti[:, pos]) & v
            pos_corr[pos] += int(ok.sum()); pos_tot[pos] += int(v.sum())
            if al is not None:
                for a, isv, hit in zip(al[i:i+bs], v.cpu().numpy(), ok.cpu().numpy()):
                    if isv: al_tot[a] += 1; al_corr[a] += int(hit)
    per_pos = pd.DataFrame({"position": np.arange(max_len)+1,
                            "acc": pos_corr/np.clip(pos_tot, 1, None), "n": pos_tot.astype(int)})
    per_pos = per_pos[per_pos.n > 0].reset_index(drop=True)
    per_al = None
    if al is not None:
        per_al = (pd.DataFrame([{"allele": a, "acc": al_corr[a]/al_tot[a], "n": al_tot[a]} for a in al_tot])
                  .sort_values("n", ascending=False).reset_index(drop=True))
    return per_pos, per_al


# ----------------------------------------------------------------------------
# Conditional pose-VAE: two training schemes (to test if pretraining helps)
#
#   Scheme 1  pretrain-then-finetune:  pretrain_cond_pose_vae (self-supervised,
#             towers frozen) -> warm-start cpose -> train_panfusion(aux=True)
#   Scheme 2  joint:                   train_panfusion(aux=True) directly
#
# Both assume the TCR / peptide / MHC towers are already warm-started
# (make_warm_start with tcr_enc, pep_state, mhc_state).
# ----------------------------------------------------------------------------
def pan_loader(frame, trans_scale, esm_table, bs=64, shuffle=True, mhc_col="mhc_seq"):
    """DataLoader over PanPairData (TCR chains + peptide + ESM-2 MHC + raw pose + mask)."""
    ds = PanPairData(frame.reset_index(drop=True), trans_scale, esm_table, mhc_col=mhc_col)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def set_pose_only_trainable(model):
    """Freeze everything except the conditional pose VAE (model.cpose)."""
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("cpose")


def unfreeze(model):
    """Re-enable grads on all parameters (call before joint fine-tuning)."""
    for p in model.parameters():
        p.requires_grad = True


def pretrain_cond_pose_vae(model, loader, epochs=40, lr=1e-3, beta=0.1, log=True,
                           return_history=False, early_stop=True, patience=8, min_delta=1e-4):
    """SCHEME 1 -- self-supervised conditional pose-VAE pretraining.

    With the TCR/peptide/MHC towers frozen (already warm-started), train ONLY
    `model.cpose` to reconstruct pose given cond=[Va,Vb,p,h], masked to structure-
    present rows (label-agnostic; uses every posed example in `loader`). Returns the
    trained `model.cpose` state_dict (+ optional history) so it can warm-start the
    classifier's pose tower. Towers run in eval mode (no BN/dropout drift); the VAE
    runs in train mode (latent sampling on). Plateau early-stop on recon+beta*KL.
    """
    import pandas as pd
    if not getattr(model, "use_pose", False):
        raise ValueError("model has no pose tower (use_pose=False)")
    set_pose_only_trainable(model)
    model.to(DEVICE).eval(); model.cpose.train()        # freeze towers, sample in VAE
    opt = torch.optim.Adam(model.cpose.parameters(), lr=lr)
    hist, best, wait, best_state, best_ep = [], float("inf"), 0, None, 0
    for ep in range(1, epochs+1):
        tot = n = 0
        for b, _ in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            _, recon, mu, lv = model(b, return_aux=True)
            loss = cvae_loss_masked(recon, b["pose"], mu, lv, b["pose_mask"], beta)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = b["pose"].size(0); tot += float(loss)*bs; n += bs
        ep_loss = tot/max(n, 1); hist.append(dict(epoch=ep, loss=ep_loss))
        if log: print(f"[cpose-pretrain] ep {ep:3d}  recon+beta*KL {ep_loss:.4f}")
        if ep_loss < best - min_delta:
            best, wait, best_ep = ep_loss, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.cpose.state_dict().items()}
        else:
            wait += 1
            if early_stop and wait >= patience:
                if log: print(f"[cpose-pretrain] early stop ep {ep} (best ep {best_ep}, {best:.4f})")
                break
    if best_state is not None:
        model.cpose.load_state_dict(best_state)
    state = {k: v.detach().cpu().clone() for k, v in model.cpose.state_dict().items()}
    return (state, pd.DataFrame(hist)) if return_history else state


TOWER_PREFIXES = ("encoders.", "pep_encoder", "mhc_encoder")   # the pretrained towers


def _apply_freeze(model, freeze):
    """Set requires_grad=False on params under any name-prefix in `freeze`, and put the
    corresponding (frozen) submodules in eval() so BatchNorm running stats don't drift."""
    freeze = (freeze,) if isinstance(freeze, str) else tuple(freeze)
    for n, p in model.named_parameters():
        p.requires_grad = not any(n.startswith(f) for f in freeze)
    model.train()                                   # default: trainable parts in train mode
    if freeze:
        for name, mod in model.named_modules():
            if name and any(name.startswith(f) for f in freeze):
                mod.eval()                          # frozen tower -> eval (no BN/dropout updates)
    return freeze


def train_panfusion(model, loader, epochs=15, lr=1e-3, pw=1.0, aux=True,
                    lam_pose=0.3, beta=0.1, freeze=(), encoder_lr=None,
                    encoder_prefixes=TOWER_PREFIXES, weight_decay=1e-4):
    """SCHEME 2 (or the fine-tune stage of Scheme 1) -- supervised classification,
    with freeze / discriminative-LR control for the freeze-vs-finetune ablation.

    freeze        : name-prefix(es) to freeze, e.g.
                    ("pep_encoder","mhc_encoder","encoders.va","encoders.vb").
                    Frozen towers are also put in eval() (no BatchNorm drift).
    encoder_lr    : if set, tower params (matching `encoder_prefixes`) train at this LR
                    while head/pose train at `lr` (discriminative / gradual unfreezing).
                    None -> single LR for all trainable params.
    aux           : add lam_pose*recon + beta*KL (conditional pose VAE trained jointly).

    Ablation presets:
      all-frozen towers   : freeze=TOWER_PREFIXES
      TCR-finetune only   : freeze=("pep_encoder","mhc_encoder")
      discriminative LR   : encoder_lr=lr/10   (no freeze)
      full fine-tune      : defaults (freeze=(), encoder_lr=None)
    """
    model.to(DEVICE)
    _apply_freeze(model, freeze)
    is_enc = lambda n: any(n.startswith(pre) for pre in encoder_prefixes)
    if encoder_lr is None:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    else:
        enc  = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
        rest = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
        groups = [{"params": enc, "lr": encoder_lr}, {"params": rest, "lr": lr}]
    groups = [g for g in groups if g["params"]]      # drop empty groups (Adam errors on empty)
    opt = torch.optim.Adam(groups, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))
    use_aux = aux and getattr(model, "use_pose", False)
    for _ in range(epochs):
        _apply_freeze(model, freeze)                 # re-assert train/eval modes each epoch
        for b, y in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}; y = y.to(DEVICE)
            if use_aux:
                logit, recon, mu, lv = model(b, return_aux=True)
                loss = crit(logit, y) + lam_pose*cvae_loss_masked(recon, b["pose"], mu, lv, b["pose_mask"], beta)
            else:
                loss = crit(model(b), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ----------------------------------------------------------------------------
# Cached training for the FROZEN arms (towers run once, not every epoch)
#
# The encoders dominate cost; when encoders+pep_encoder+mhc_encoder are frozen their
# outputs are deterministic, so precompute them once and train only the cheap trainable
# parts (tcr_proj + cpose + head). Turns N_epochs of encoder forwards into 1.
# Use for: all-frozen-towers arm, the frozen-probe diagnostic, and Scheme-1 pretraining.
# NOT valid once any of those towers is unfrozen (e.g. discriminative-LR / full fine-tune).
# ----------------------------------------------------------------------------
@torch.no_grad()
def cache_tower_features(model, loader):
    """Run the frozen towers once over `loader` (PanPairData) and cache their outputs as
    CPU tensors: per-chain encoder embeddings, peptide p, MHC h, raw pose, mask, label.
    Assumes encoders / pep_encoder / mhc_encoder are the frozen towers."""
    model.to(DEVICE).eval()
    chains = list(model.chains)
    buf = {c: [] for c in chains}; buf.update(p=[], h=[], pose=[], pose_mask=[], y=[])
    for b, y in loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        enc = {c: model.encoders[c](b[c]) for c in chains}
        h = model.mhc_encoder(b["mhc"]); p = model.pep_encoder(b["pep"], h)
        for c in chains: buf[c].append(enc[c].cpu())
        buf["p"].append(p.cpu()); buf["h"].append(h.cpu())
        buf["pose"].append(b["pose"].cpu()); buf["pose_mask"].append(b["pose_mask"].cpu())
        buf["y"].append(y)
    return {k: torch.cat(v) for k, v in buf.items()}


def _cached_loader(cache, fields, bs, shuffle):
    ds = TensorDataset(*[cache[k] for k in fields])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train_cached(model, cache, epochs=15, lr=1e-3, pw=1.0, aux=True, lam_pose=0.3,
                 beta=0.1, bs=512, weight_decay=1e-4, shuffle=True):
    """Joint classification (+ optional aux VAE) on cached frozen-tower features.
    Trains only tcr_proj + cpose + head (+ null_pose); encoders/pep/mhc stay frozen.
    Much cheaper than train_panfusion(freeze=TOWER_PREFIXES) because sequences are never
    re-encoded. Respects model.head_feats. Build `cache` with cache_tower_features."""
    model.to(DEVICE)
    _apply_freeze(model, TOWER_PREFIXES)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))
    chains = list(model.chains)
    fields = chains + ["p", "h", "pose", "pose_mask", "y"]
    dl = _cached_loader(cache, fields, bs, shuffle)
    use_aux = aux and getattr(model, "use_pose", False)
    for _ in range(epochs):
        for batch in dl:
            v = {k: t.to(DEVICE) for k, t in zip(fields, batch)}
            enc = {c: v[c] for c in chains}
            t = model.tcr_proj(torch.cat([enc[c] for c in chains], 1))
            p, h, pose, mask, y = v["p"], v["h"], v["pose"], v["pose_mask"], v["y"]
            recon = mu = lv = None; pose_mu = None
            if model.use_pose:
                cond = torch.cat([enc["va"], enc["vb"], p, h], 1)
                recon, mu, lv, _ = model.cpose(pose, cond)
                mk = mask.float().unsqueeze(1); pose_mu = mk*mu + (1-mk)*model.null_pose.unsqueeze(0)
            fmap = {"t": t, "p": p, "h": h, "tp": t*p, "th": t*h, "pose": pose_mu}
            logit = model.head(torch.cat([fmap[f] for f in model.head_feats], 1)).squeeze(-1)
            loss = crit(logit, y)
            if use_aux:
                loss = loss + lam_pose*cvae_loss_masked(recon, pose, mu, lv, mask, beta)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def pretrain_cond_pose_vae_cached(model, cache, epochs=40, lr=1e-3, beta=0.1, bs=512,
                                  log=True, return_history=False, early_stop=True,
                                  patience=8, min_delta=1e-4):
    """SCHEME 1 on cached features: train ONLY model.cpose to reconstruct pose given
    cond=[Va,Vb,p,h] (all cached). Returns cpose state_dict (+ history)."""
    import pandas as pd
    if not getattr(model, "use_pose", False):
        raise ValueError("model has no pose tower (use_pose=False)")
    model.to(DEVICE); set_pose_only_trainable(model); model.eval(); model.cpose.train()
    opt = torch.optim.Adam(model.cpose.parameters(), lr=lr)
    fields = ["va", "vb", "p", "h", "pose", "pose_mask"]
    dl = _cached_loader(cache, fields, bs, shuffle=True)
    hist, best, wait, best_state, best_ep = [], float("inf"), 0, None, 0
    for ep in range(1, epochs+1):
        tot = n = 0
        for batch in dl:
            v = {k: t.to(DEVICE) for k, t in zip(fields, batch)}
            cond = torch.cat([v["va"], v["vb"], v["p"], v["h"]], 1)
            recon, mu, lv, _ = model.cpose(v["pose"], cond)
            loss = cvae_loss_masked(recon, v["pose"], mu, lv, v["pose_mask"], beta)
            opt.zero_grad(); loss.backward(); opt.step()
            b_ = v["pose"].size(0); tot += float(loss)*b_; n += b_
        ep_loss = tot/max(n, 1); hist.append(dict(epoch=ep, loss=ep_loss))
        if log: print(f"[cpose-pretrain-cached] ep {ep:3d}  recon+beta*KL {ep_loss:.4f}")
        if ep_loss < best - min_delta:
            best, wait, best_ep = ep_loss, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.cpose.state_dict().items()}
        else:
            wait += 1
            if early_stop and wait >= patience:
                if log: print(f"[cpose-pretrain-cached] early stop ep {ep} (best ep {best_ep}, {best:.4f})")
                break
    if best_state is not None:
        model.cpose.load_state_dict(best_state)
    state = {k: v.detach().cpu().clone() for k, v in model.cpose.state_dict().items()}
    return (state, pd.DataFrame(hist)) if return_history else state
