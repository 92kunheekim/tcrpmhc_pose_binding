#!/usr/bin/env python3
"""Gather a vast, deduplicated TCR-chain corpus from all data/ sources for
self-supervised pretraining of the per-chain encoders (va, vb, cdr3a, cdr3b).

Each chain encoder is an independent autoencoder, so chains are deduplicated
*independently* (sharing a chain across many pairs would otherwise over-weight
common chains). Full chains are split into (CDR3 core, V-region prefix) with the
same `split_chain` the model uses, then filtered to valid amino-acid strings.

Output (data/tcr_pretraining/):
  tcr_pretrain_corpus.csv   wide: va, vb, cdr3a, cdr3b (independently deduped, ""-padded)
  source_manifest.csv       per-source contribution counts
  README.md                 summary + how to use
"""
import os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data"))   # project data/

AA = set("ACDEFGHIKLMNPQRSTVWY")
_VALID = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# torch-free copy of src/data.py::split_chain (IMGT-junction CDR3 + V-region prefix)
_PAT = re.compile(r"C[A-Z]{4,28}?[FW]G[A-Z]G")
def split_chain(chain):
    if not isinstance(chain, str):
        return "", ""
    h = list(_PAT.finditer(chain))
    if not h:
        return "", chain[:110]
    m = h[-1]
    return m.group()[1:-4], chain[:m.start()]

# full-chain sources -> (path, alpha full col, beta full col, source name)
FULL_SOURCES = [
    ("full_seq_df_new.csv",                         "TCR_A_sequence", "TCR_B_sequence", "full_seq_df"),
    ("final_dataset_modeled_vj.csv",                "TCR_A_sequence", "TCR_B_sequence", "final_modeled_vj"),
    ("raw/combined_positives.csv",                  "TRA_full_seq",   "TRB_full_seq",   "combined_positives"),
    ("raw/combined_nonbinders.csv",                 "TRA_full_seq",   "TRB_full_seq",   "combined_nonbinders"),
    ("raw/VDJdb_classI_paired_filtered.csv",        "TRA_full_seq",   "TRB_full_seq",   "vdjdb_classI"),
    ("raw/IEDB_positive_filtered.csv",              "TRA_protein_seq","TRB_protein_seq","iedb_positive"),
    ("10x_cd8_A0201_6pep/inputs/combined_sorted.csv","tcra_seq",      "tcrb_seq",       "10x_cd8_A0201"),
]

# CDR3-only sources -> (path, alpha cdr3 col, beta cdr3 col, source name)
CDR3_SOURCES = [
    ("raw/McPAS-TCR.csv",                       "CDR3.alpha.aa", "CDR3.beta.aa", "mcpas"),
    ("modeled_vj_cdr3_by_chain.csv",            "CDR3a",         "CDR3b",        "modeled_vj_cdr3"),
    ("raw/decoys_permuted.csv",                 "CDR3a",         "CDR3b",        "decoys_permuted"),
    ("raw/10X_CD8T_donors/all_donors_binders.csv",    "CDR3a",   "CDR3b",        "10x_donors_binders"),
    ("raw/10X_CD8T_donors/all_donors_nonbinders.csv", "CDR3a",   "CDR3b",        "10x_donors_nonbinders"),
]

# length sanity bounds
V_MIN, V_MAX = 40, 130
C_MIN, C_MAX = 4, 30


def clean(seq):
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper().replace("*", "").replace(" ", "")
    return s if s and _VALID.match(s) else None


def norm_cdr3(s):
    """Normalize a CDR3 to the model's split_chain core convention: strip a leading
    conserved C and trailing conserved F/W (IMGT junction form -> core). Length-bounded."""
    s = clean(s)
    if s is None:
        return None
    if len(s) >= 3 and s[0] == "C" and s[-1] in "FW":
        s = s[1:-1]
    return s if C_MIN <= len(s) <= C_MAX else None


def parts_from_full(full):
    """full chain -> (cdr3_core, v_prefix) keeping only valid, length-bounded parts."""
    s = clean(full)
    if s is None:
        return None, None
    cdr3, v = split_chain(s)
    cdr3 = norm_cdr3(cdr3)
    v = v if (V_MIN <= len(v) <= V_MAX) else None
    return cdr3, v


def main():
    # per-chain unique sets + per-source/per-chain tallies
    uniq = {"va": set(), "vb": set(), "cdr3a": set(), "cdr3b": set()}
    manifest = []

    def tally(name, rel, nrows):
        row = {"source": name, "file": rel, "rows": nrows}
        for k in uniq:
            row[f"new_{k}"] = len(uniq[k]) - before[k]
        manifest.append(row)
        print(f"  {name:22s} rows={nrows:>7}  "
              + " ".join(f"+{k}:{row[f'new_{k}']}" for k in uniq))

    # 1. full chains -> va/vb + cdr3a/cdr3b
    for rel, acol, bcol, name in FULL_SOURCES:
        path = os.path.join(DATA, rel)
        if not os.path.exists(path):
            print(f"  skip (missing): {rel}"); continue
        want = {c for c in (acol, bcol) if c}
        df = pd.read_csv(path, usecols=lambda c, w=want: c in w, low_memory=False)
        before = {k: len(v) for k, v in uniq.items()}
        for col, vchain, cchain in [(acol, "va", "cdr3a"), (bcol, "vb", "cdr3b")]:
            if col not in df.columns:
                continue
            for full in df[col].values:
                cdr3, v = parts_from_full(full)
                if v:    uniq[vchain].add(v)
                if cdr3: uniq[cchain].add(cdr3)
        tally(name, rel, len(df))

    # 2. CDR3-only sources -> cdr3a/cdr3b (normalized to the same core convention)
    for rel, acol, bcol, name in CDR3_SOURCES:
        path = os.path.join(DATA, rel)
        if not os.path.exists(path):
            print(f"  skip (missing): {rel}"); continue
        want = {acol, bcol}
        df = pd.read_csv(path, usecols=lambda c, w=want: c in w, low_memory=False)
        before = {k: len(v) for k, v in uniq.items()}
        for col, cchain in [(acol, "cdr3a"), (bcol, "cdr3b")]:
            if col not in df.columns:
                continue
            for raw in df[col].values:
                c = norm_cdr3(raw)
                if c: uniq[cchain].add(c)
        tally(name, rel, len(df))

    # 3. raw VDJDB.tsv (Gene column = TRA/TRB, single CDR3 column)
    vpath = os.path.join(DATA, "raw/VDJDB.tsv")
    if os.path.exists(vpath):
        df = pd.read_csv(vpath, sep="\t", usecols=lambda c: c in {"Gene", "CDR3"}, low_memory=False)
        before = {k: len(v) for k, v in uniq.items()}
        for gene, raw in zip(df.get("Gene", []), df.get("CDR3", [])):
            c = norm_cdr3(raw)
            if not c:
                continue
            if gene == "TRA":   uniq["cdr3a"].add(c)
            elif gene == "TRB": uniq["cdr3b"].add(c)
        tally("vdjdb_raw", "raw/VDJDB.tsv", len(df))

    # wide, independently-deduped, ""-padded corpus
    cols = {k: sorted(uniq[k]) for k in ("va", "vb", "cdr3a", "cdr3b")}
    n = max(len(v) for v in cols.values())
    corpus = pd.DataFrame({k: v + [""] * (n - len(v)) for k, v in cols.items()})

    out_dir = HERE
    corpus.to_csv(os.path.join(out_dir, "tcr_pretrain_corpus.csv"), index=False)
    pd.DataFrame(manifest).to_csv(os.path.join(out_dir, "source_manifest.csv"), index=False)

    counts = {k: len(v) for k, v in cols.items()}
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("# TCR pretraining corpus\n\n")
        fh.write("Deduplicated TCR chains for self-supervised pretraining of the per-chain "
                 "encoders (`pretrain_tcr_encoders`). Each chain is deduplicated **independently** "
                 "(the four autoencoders train separately), so columns are *not* row-aligned and "
                 "shorter columns are padded with empty strings.\n\n")
        fh.write("## Unique chains\n\n")
        for k in ("va", "vb", "cdr3a", "cdr3b"):
            fh.write(f"- **{k}**: {counts[k]:,}\n")
        fh.write(f"\nTotal unique chains: {sum(counts.values()):,}\n\n")
        fh.write("## Sources\n\nSee `source_manifest.csv` (rows + new-unique contribution per chain).\n\n")
        fh.write("## Use\n\n```python\nimport pandas as pd\nfrom train_utils import pretrain_tcr_encoders\n"
                 "seq = pd.read_csv('data/tcr_pretraining/tcr_pretrain_corpus.csv').fillna('')\n"
                 "TCR_ENC, hist = pretrain_tcr_encoders(seq, return_history=True)\n```\n"
                 "`pretrain_tcr_encoders` drops empty/padding entries per chain automatically.\n")

    print("\n=== unique chains ===")
    for k in ("va", "vb", "cdr3a", "cdr3b"):
        print(f"  {k:6s} {counts[k]:>8,}")
    print(f"  TOTAL  {sum(counts.values()):>8,}")
    print(f"\nwrote -> {out_dir}/tcr_pretrain_corpus.csv  (rows={n:,})")


if __name__ == "__main__":
    main()
