# TCR pretraining corpus

Deduplicated TCR chains for self-supervised pretraining of the per-chain encoders (`pretrain_tcr_encoders`). Each chain is deduplicated **independently** (the four autoencoders train separately), so columns are *not* row-aligned and shorter columns are padded with empty strings.

## Unique chains

- **va**: 1,011
- **vb**: 504
- **cdr3a**: 58,227
- **cdr3b**: 117,260

Total unique chains: 177,002

## Sources

See `source_manifest.csv` (rows + new-unique contribution per chain).

## Use

```python
import pandas as pd
from train_utils import pretrain_tcr_encoders
seq = pd.read_csv('data/tcr_pretraining/tcr_pretrain_corpus.csv').fillna('')
TCR_ENC, hist = pretrain_tcr_encoders(seq, return_history=True)
```
`pretrain_tcr_encoders` drops empty/padding entries per chain automatically.
