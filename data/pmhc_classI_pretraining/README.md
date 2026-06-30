# peptide–MHC class I dataset (IEDB) — for peptide encoder pretraining

Source: IEDB MHC-ligand database, full export `mhc_ligand_full_tsv.zip`
(`raw/mhc_ligand_full_tsv.zip`, 289 MB → `raw/extracted/mhc_ligand_full.tsv`, 1.3 GB,
904,711 assay records).

The IEDB web UI / query API were unusably slow in this environment, so the full
MHC-ligand export was downloaded once (via browser) and filtered locally in code.
This makes the filter logic explicit and reproducible (below).

## Filters applied

Matching the requested IEDB filter set:

| Requested filter | Column (IEDB export) | Rule |
|---|---|---|
| Linear peptide | Epitope → Object Type | `== "Linear peptide"` |
| MHC assay positive outcome | Assay → Qualitative Measurement | starts with `Positive` (Positive, Positive-High, Positive-Intermediate, Positive-Low); excludes Negative |
| MHC restriction Class I | MHC Restriction → Class | `== "I"` (excludes II and "non classical") |
| Allele resolution 4-digit, 2 chains | MHC Restriction → Name | two-field allele, regex `\*\d+:\d+` (e.g. `HLA-A*02:01`); excludes serotype/locus (`HLA-A2`, `HLA-A`), single-field, and H-2-style names |
| Exclude statistically inferred evidence | MHC Restriction → Evidence Code | `!= "Statistically inferred by motif or alleles present"` |

### Funnel

```
904,711  all MHC-ligand assay records
472,792  linear + Positive* + class I + not statistically-inferred
295,677  ... AND 4-digit (two-field) allele resolution   <-- final filtered set
```

## Files

`processed/`

- **`iedb_mhc_classI_filtered.tsv`** — 295,677 rows, one per passing assay.
  Columns: `peptide, length, allele, allele_iri, mhc_class, qualitative_measure,
  evidence_code, is_standard_aa, source_organism, assay_id, epitope_iri`.
- **`iedb_mhc_classI_peptide_allele_unique.tsv`** — 230,469 unique (peptide, allele)
  pairs (standard-AA peptides only), with `n_assays` support count.
- **`iedb_mhc_classI_peptides_unique.txt`** — 167,008 unique standard-AA peptides
  (encoder-pretraining input).
- **`iedb_mhc_classI_alleles.tsv`** — 200 alleles with `n_assays`, `n_unique_peptides`.

## Notes / dataset characteristics

- **200 class I alleles**, predominantly human HLA-A/B/C; top: HLA-B*27:05 (66k rows,
  41k unique peptides), HLA-A*02:01, HLA-B*57:01.
- **Length:** dominated by 9-mers (158,579) and 10-mers (59,067); 8-mers 10,085.
  A long tail (15–50 aa) exists from elution records — if you want a strict class I
  ligand set, filter `length` to 8–11 (or 8–14).
- **Modified residues:** 5,071 of 295,677 rows carry PTM/modification annotations or
  `X` (`is_standard_aa = 0`, e.g. `GILGFVFTL + OTH(L9)`). These are kept in
  `iedb_mhc_classI_filtered.tsv` but **excluded** from the unique-peptide/peptide–allele
  files. Drop them via `is_standard_aa == 1` if not wanted.
- `qualitative_measure` is retained so you can tighten to `Positive`/`Positive-High`
  only if desired.

## MHC sequence dictionary

Protein sequences for the dataset alleles, pulled from local IMGT databases:
- `~/Documents/IMGTHLA/hla_prot.fasta` (IMGT/HLA) — human HLA-A/B/C
- `~/Documents/IMGTMHC/MHC_prot.fasta` (IPD-MHC) — non-human (Patr, Mamu, SLA,
  BoLA, Eqca, DLA, Gaga-BF, Gogo)

Built by `data/scripts/build_mhc_seq_dict.py`. Matching is at **2-field (4-digit)
resolution**: a dataset allele (e.g. `HLA-A*02:01`) maps to all IMGT entries whose
`locus*field1:field2` equals it, and the **longest** protein sequence in that group
is taken as the representative (3rd/4th allele fields are synonymous/non-coding, so
the protein is identical modulo partial sequencing).

### Files (`processed/`)

- **`mhc_classI_sequences.json`** — `{ allele: protein_sequence }`, 197 entries.
- **`mhc_classI_sequences.tsv`** — `allele, imgt_allele, source, length, sequence`.
- **`mhc_classI_sequences.fasta`** — same sequences, FASTA.

### Coverage

- **197 / 200** alleles matched (151 IMGT/HLA, 46 IPD-MHC).
- **3 unmatched** — absent from the IPD-MHC protein release at any resolution
  (≈40 of 295,677 assay rows): `SLA-3*02:02` (32 rows), `SLA-1*03:02` (4),
  `Ptal-N*01:01` (4).
- Sequences are full-length precursors (include signal peptide), ~362–372 aa for
  classical class I. **8** alleles have only a **partial** IMGT entry (181–182 aa,
  α1/α2 binding-groove region): `HLA-A*02:50, A*03:19, A*30:14L, B*08:03, B*41:05,
  B*45:06, DLA-88*501:01, DLA-88*508:01`.
- All sequences are standard amino acids (signal peptide included — trim if you
  need the mature chain).

### α1+α2 groove sequences (alignment-verified)

For the peptide-binding region only (mature residues **1–182** = α1 1–90 + α2
91–182). Built by `data/scripts/build_a1a2_alignment.py` (HLA) and
`build_a1a2_nonhuman.py` (non-human).

- **HLA (151 alleles):** extracted from the **IMGT/HLA protein alignments**
  (`A/B/C_prot.txt`, release 3.64.0, pulled from the IMGTHLA git tree). Block
  separators detected empirically so leader-length differences stay registered;
  each allele reconstructed and **verified by exact match to `hla_prot.fasta`**
  — 19,335 / 19,335 reconstructed sequences matched (0 mismatches). All 151
  dataset HLA alleles give exactly 182 aa.
- **Non-human (46 alleles):** no IMGT alignment ships for IPD-MHC, so the groove
  is taken by local alignment (BLOSUM62) of the full IPD-MHC protein against the
  verified HLA-A*02:01 α1+α2 reference. `pct_id_to_HLA_ref` is reported per
  allele: chimp Patr-A 88–92%, macaque/cow/pig/dog/horse/gorilla 71–83%,
  chicken Gaga-BF ~47–51% (divergent — use with care). Lengths 176–182, except
  `Eqca-1*002:01` = 148 aa (partial IPD-MHC entry).

Files (`processed/`):
- `mhc_classI_a1a2.json` — `{ allele: a1a2_sequence }`, 197 entries
- `mhc_classI_a1a2.tsv` — `allele, imgt_allele, source, a1a2_length,
  pct_id_to_HLA_ref, a1a2_sequence` (`source` = `IMGT/HLA-align` vs
  `IPD-MHC-homology`)
- `mhc_classI_a1a2.fasta`
- `a1a2_alignment_verification.txt` — QC report

This α1+α2 region corresponds to the groove platform used in `frame_pose.py`
(`calc_mhc_vectors`, class I Cα residues 1–150) — i.e. the same domains that
build the pMHC pose frame, but as full 1–182 sequence rather than the 1–150
structural cut.

## Reproduce

Filter command lives in the session history; core predicate (awk over the export):

```
Object Type == "Linear peptide"
AND MHC Class == "I"
AND Qualitative ~ /^Positive/
AND Evidence Code != "Statistically inferred by motif or alleles present"
AND MHC Name ~ /\*[0-9]+:[0-9]+/
```
