# tcrpmhc_pose_binding

TCR–pMHC binding prediction from TCR **sequence** + FramePose **docking geometry**.
Three model families, sharing a common backbone:

- **peptide-specific** (`src/peptide_specific.py`) — fixed antigen, evaluated per peptide;
  no peptide/MHC tower. Models: `SeqOnly`, `PoseOnlyClf`, `TCRPoseFusion` (+ tree baselines).
- **across-peptide** (`src/across_peptide.py`) — adds a peptide tower; MHC still fixed.
  Models: `PeptideAwareFusion`, `CondPeptideAware(Fusion)` (V/peptide-conditioned pose).
  Evaluated by MIXED (novel TCR) and LOPO (unseen peptide).
- **pan-specific** (`src/pan_specific.py`) — *future*: adds an MHC tower (ESM-2) to
  generalize across HLA alleles. Scaffold/stub only.

## Layout
```
src/
  config.py        # BODIES (7), CHAINS, padding, raw_columns
  atchley.py       # Atchley encoding
  geometry.py      # SH(S2), quaternion<->matrix, 6D rep, geodesic, QuaternionLinear
  encoders.py      # ChainEncoder, ChainAutoencoder (warm-start)
  pose_vae.py      # PoseVAERaw (unconditioned) + vae_raw_loss
  pose_cvae.py     # ConditionalPoseVAERaw + cvae_loss_masked
  data.py          # load_data, cluster_tcrs, repeated_splits, add_mismatch, PairData, tree features
  train_utils.py   # pretrain (TCR AE, pose VAE), train_one(_aux), run_repeated, cluster_bootstrap
  peptide_specific.py
  across_peptide.py
  pan_specific.py  # FUTURE (MHC-aware)
notebooks/  # Colab notebook (Google Drive I/O)
data/       # inputs: combined_sorted.csv, pose_descriptors.csv, iptm_table.csv
results/    # tables + figures
```

## Conventions
- 7 FramePose bodies: TCR + CDR1/2/3 of α and β.
- Evaluation: 10× repeated leakage-controlled 80/20 splits (V-gene+CDR3 clusters held whole);
  per-sample-aggregated predictions; cluster bootstrap for 95% CIs.
- ipTM ≥ 0.5 structure filter; pose-discrimination uses posed examples only
  (mismatch negatives off — they have null pose and would leak as a label proxy).

## Run (with src on path)
```python
import sys; sys.path.insert(0, "src")
from data import load_data
from train_utils import pretrain_tcr_encoders, make_warm_start, pretrain_posevae
from peptide_specific import run_per_peptide
from across_peptide import PeptideAwareFusion, CondPeptideAwareFusion, run_mixed, run_lopo

D = load_data("data", min_iptm=0.5)
warm = make_warm_start(pretrain_tcr_encoders(D["pool"]))
vae6 = pretrain_posevae(D["POSE_ALL"], "6d"); vaeq = pretrain_posevae(D["POSE_ALL"], "qnn")
resA = run_per_peptide(D["pool"], D["trans_scale"], vae6, vaeq, warm)
```
