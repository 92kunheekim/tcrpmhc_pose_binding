# tcrpmhc_pose_binding

TCR–pMHC binding prediction from TCR **sequence** + FramePose **docking geometry**,
extended toward a **pan-allele** (class I) model with an MHC tower. Assesses whether
predicted-structure-derived signal (docking pose + confidence) reliably separates
binders from non-binders and adds value beyond sequence.

## Model families

All share a per-chain sequence backbone (Vα, Vβ, CDR3α, CDR3β encoders).

- **peptide-specific** (`src/peptide_specific.py`) — fixed antigen, evaluated per peptide;
  no peptide/MHC tower. `SeqOnly`, `PoseOnlyClf`, `TCRPoseFusion` (+ tree baselines).
- **across-peptide** (`src/across_peptide.py`) — adds a peptide tower; MHC fixed.
  `PeptideAwareFusion`, `CondPeptideAware(Fusion)` (V/peptide-conditioned pose).
  Evaluated by MIXED (novel TCR) and LOPO (unseen peptide).
- **pan-specific** (`src/pan_specific.py`) — **built**. Adds an ESM-2 MHC tower and an
  MHC-conditioned peptide tower; the pose VAE is conditioned on **[TCR, peptide, MHC]**.
  Class I, α1α2 groove. Note: predicted structures currently exist only for the
  single-allele A\*02:01 set, so the MHC channel is wired but *dormant* until multi-allele
  structures are added — the peptide pretraining, however, uses multi-allele IEDB data, so
  its MHC-conditioning is exercised.

## pan-specific architecture (`PanFusion`)

| tower | input | encoder | dim |
|---|---|---|---|
| TCR | 4 chains (Atchley) | `ChainEncoder` ×4 → `tcr_proj` | t = 64 |
| peptide | peptide (Atchley) | `PeptideEncoderTransformer` (FiLM on MHC) / `MHCConditionedPeptideEncoder` | p = 64 |
| MHC | ESM-2 of α1α2 groove (precomputed) | `MHCEncoder` (1280→64) | h = 64 |
| pose | 7 bodies × (trans+quat) | `ConditionalPoseVAERaw`, cond=[Vα,Vβ,p,h] | pose_mu = 4·7 |

- **Head feature presets** (`head_feats`): `"pose"`, `"pose_tp"`, `"pose_tp_th"`, `"full"`
  (blocks from `{t, p, h, tp=t*p, th=t*h, pose}`) — for isolating pose vs sequence.
- `pep_arch="transformer"` (per-position MLM-pretrainable) or `"conv"` (pooled).
- `use_pose=False` → seq+pep+MHC baseline. `th=t*h` is collinear with `t` under one
  allele — include only with multi-allele data.

## Pretraining

- **TCR encoders** (`pretrain_tcr_encoders`) — per-chain Atchley autoencoder (MSE) on a
  177k-chain corpus (`data/tcr_pretraining/`). Warm-start / regularizer; scaffold-heavy.
- **Peptide encoder** (`pretrain_peptide_masked`, `arch="transformer"`) — self-supervised
  **MHC-conditioned masked-residue** MLM on IEDB class-I `(peptide, allele)` pairs
  (`data/pmhc_classI_pretraining/`); learns allele-specific anchor motifs. Diagnose with
  `masked_accuracy_breakdown` (per-position + per-allele; look for anchor spikes).
- **Pose VAE** (`pretrain_posevae`) — unconditioned, frozen after pretraining.
- **Conditional pose VAE** — two schemes to test if pretraining helps:
  1. *pretrain-then-finetune*: `pretrain_cond_pose_vae` (towers frozen) → `train_panfusion`.
  2. *joint*: `train_panfusion(aux=True)` (`BCE + λ·recon + β·KL`).
  `train_panfusion` supports `freeze=` / `encoder_lr=` (freeze vs discriminative-LR ablation).
  **Cached path** for frozen arms (encode towers once): `cache_tower_features` →
  `train_cached` / `pretrain_cond_pose_vae_cached`.

## Layout
```
src/
  config.py        # BODIES (7), CHAINS, padding, raw_columns
  atchley.py geometry.py        # encoding; SH(S2), quat<->matrix, 6D rep, geodesic, QuaternionLinear
  encoders.py      # ChainEncoder, ChainAutoencoder, SeqVAE
  pose_vae.py pose_cvae.py      # PoseVAERaw / ConditionalPoseVAERaw + losses
  data.py          # load_data, cluster_tcrs, repeated_splits, PairData, tree features
  train_utils.py   # pretrain (TCR AE, pose VAE), make_warm_start, train_one(_aux), run_repeated, cluster_bootstrap
  peptide_specific.py across_peptide.py
  pan_specific.py  # MHC/ESM-2, transformer peptide MLM, PanFusion, schemes, cached training, diagnostics
  analysis.py      # latent / reconstruction comparison (binder vs non-binder VAEs)
notebooks/         # Colab (clone from GitHub, push results) — see below
data/
  10x_cd8_A0201_6pep/ ...        # inputs: combined_sorted.csv, pose_descriptors.csv, iptm_table.csv
  tcr_pretraining/               # tcr_pretrain_corpus.csv (177k chains) + build_corpus.py
  pmhc_classI_pretraining/       # IEDB peptide-allele TSV + α1α2 groove JSON
pretrained/        # saved encoders / VAEs + curves (committed from Colab)
results/           # tables + figures
```

## Notebooks (Colab)
- `pretrain_tcr.ipynb` — TCR chain encoders on the corpus.
- `pretrain_peptide.ipynb` — MHC-conditioned masked-residue peptide transformer MLM on IEDB
  + per-position/per-allele accuracy diagnostic.
- `pretrain.ipynb` — pose VAEs (both strategies) + TCR encoders.
- `analyze_latent.ipynb` — binder-only vs non-binder-only pose-VAE latent comparison.
- `train.ipynb` — training driver (peptide-specific + across-peptide).

Each clones the repo into `/content` (token from Colab **Secrets** `GITHUB_TOKEN`),
writes outputs into `pretrained/`, and commits + pushes. Existing clones are `git pull`-ed
to latest `main`.

## Conventions
- 7 FramePose bodies: TCR + CDR1/2/3 of α and β.
- ipTM ≥ 0.5 structure filter; pose-discrimination uses posed examples only (mismatch
  negatives off — null pose would leak as a label proxy).
- Evaluation: leakage-controlled splits (V-gene+CDR3 clusters held whole), per-sample-
  aggregated predictions, cluster bootstrap 95% CIs, AUPRC vs prevalence baseline.
  Priority held-out axes: **LOPO** (unseen peptide) and **TCR clusters** (unseen receptor).
- Negative controls: pose/sequence permutation, ipTM-only, peptide-stratified pose-only.

## Quick start (local, `src` on path)
```python
import sys; sys.path.insert(0, "src")
from data import load_data
from train_utils import pretrain_tcr_encoders, make_warm_start, pretrain_posevae
from peptide_specific import run_per_peptide

D = load_data("data", min_iptm=0.5)
warm = make_warm_start(pretrain_tcr_encoders(D["pool"]))
vae6 = pretrain_posevae(D["POSE_ALL"], "6d"); vaeq = pretrain_posevae(D["POSE_ALL"], "qnn")
resA = run_per_peptide(D["pool"], D["trans_scale"], vae6, vaeq, warm)
```

### Pan model (warm-started towers + cached frozen training)
```python
import torch
from pan_specific import PanFusion, pan_loader, cache_tower_features, train_cached
from train_utils import make_warm_start

TCR_ENC   = torch.load("pretrained/tcr_encoders_corpus.pt")
pep_state = torch.load("pretrained/pep_encoder_masked.pt")
mhc_state = torch.load("pretrained/mhc_encoder_masked.pt")
warm  = make_warm_start(TCR_ENC, pep_state=pep_state, mhc_state=mhc_state)
model = warm(PanFusion(use_pose=True, pep_arch="transformer", head_feats="full"))

cache = cache_tower_features(model, pan_loader(train_df, trans_scale, esm_table))  # towers encoded once
train_cached(model, cache, aux=True)                                              # frozen-towers training
```
Use the same α1α2 ESM-2 region for `esm_table` as the peptide pretraining, so the
warm-started `mhc_encoder` sees a consistent input distribution.
