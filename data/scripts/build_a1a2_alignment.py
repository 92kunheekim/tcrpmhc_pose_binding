#!/usr/bin/env python3
"""Extract and VERIFY the class I alpha1+alpha2 (groove) sequence per allele
using the IMGT/HLA protein ALIGNMENTS (A/B/C_prot.txt).

Method
------
- IMGT _prot.txt is a column alignment. Residue field starts at column 19;
  mature residue 1 is at column 54 (page 1). Residues are printed in blocks of
  10 separated by single spaces.
- Block-separator columns are detected EMPIRICALLY (columns blank in *every*
  allele row), so leader length differences stay positionally registered.
- The first data row of page 1 is the reference; '-' = identical to reference,
  '.' = gap/indel, '*' = unknown, letters = substitutions.
- Each allele is reconstructed (apply reference at '-'), de-gapped, and the
  full sequence is checked against hla_prot.fasta (verification).
- alpha1+alpha2 = mature residues 1..182 (window defined by the reference's
  residue numbering, so it is consistent across alleles).

Outputs (processed/):
  mhc_classI_a1a2.json / .tsv / .fasta   (dataset alleles)
  a1a2_alignment_verification.txt        (QC report)
"""
import json, os, re, glob

HLA_DIR = "/sessions/eager-sharp-pasteur/mnt/IMGTHLA"          # git repo (alignments in HEAD)
HLA_FASTA = f"{HLA_DIR}/hla_prot.fasta"
ALIGN_TMP = "/tmp"                                              # A_prot.txt etc. extracted here
ALLELE_TSV = ("/sessions/eager-sharp-pasteur/mnt/tcrpmhc_model_structure_reliability/"
              "tcrpmhc_pose_binding/data/pmhc_classI_pretraining/processed/"
              "iedb_mhc_classI_alleles.tsv")
OUTDIR = ("/sessions/eager-sharp-pasteur/mnt/tcrpmhc_model_structure_reliability/"
          "tcrpmhc_pose_binding/data/pmhc_classI_pretraining/processed")

SC = 19            # residue field start column (0-based)
MATURE1_COL = 54   # absolute column of mature residue 1 on page 1
A1A2_LEN = 182     # mature residues 1..182 = alpha1 (1-90) + alpha2 (91-182)


def two_field(name):
    if "*" not in name:
        return None
    loc, rest = name.split("*", 1)
    f = rest.split(":")
    if len(f) < 2:
        return None
    return f"{loc}*{f[0]}:{re.sub(r'[A-Za-z]+$','',f[1])}"


def parse_fasta(path):
    d, name, seq = {}, None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name:
                d[name] = "".join(seq)
            p = line[1:].split()
            name = p[1] if len(p) > 1 else p[0]
            seq = []
        else:
            seq.append(line.strip())
    if name:
        d[name] = "".join(seq)
    return d


def parse_alignment(path):
    """Return ordered list of allele names and dict name->reconstructed full seq,
    plus name->alpha1a2. Reference = first data row of page 1."""
    lines = open(path).read().split("\n")
    # split into pages by ' Prot' header
    pages, cur = [], None
    for ln in lines:
        if ln.startswith(" Prot"):
            cur = []
            pages.append(cur)
        elif cur is not None:
            toks = ln.split()
            if toks and "*" in toks[0]:
                cur.append(ln)
    # per page: build name->raw residue-field string (from SC), detect separator cols
    order = []
    aligned = {}       # name -> list of aligned columns (across pages)
    mature1_idx = None  # index within page-0 kept columns
    for pi, rows in enumerate(pages):
        width = max(len(r) for r in rows)
        rows = [r.ljust(width) for r in rows]
        names = [r.split()[0] for r in rows]
        # separator columns: blank in every row, for col >= SC
        keep_cols = []
        for c in range(SC, width):
            if any(r[c] != " " for r in rows):
                keep_cols.append(c)
        if pi == 0:
            # index (within keep_cols) of first kept col >= MATURE1_COL
            mature1_idx = sum(1 for c in keep_cols if c < MATURE1_COL)
        for r, nm in zip(rows, names):
            cols = [r[c] for c in keep_cols]
            aligned.setdefault(nm, [])
            aligned[nm].extend(cols)
            if pi == 0 and nm not in order:
                order.append(nm)
    # also append later-page names already handled via setdefault/extend
    ref_name = order[0]
    ref_cols = aligned[ref_name]
    n = len(ref_cols)

    def resolve(nm):
        cols = aligned[nm]
        cols = cols + [" "] * (n - len(cols))
        out = []
        for i, ch in enumerate(cols):
            if ch == "-":
                out.append(ref_cols[i])
            else:
                out.append(ch)
        return out

    full, a1a2 = {}, {}
    # reference residue numbering window for mature 1..182
    # walk ref columns from mature1_idx, counting ref residues (non '.', non ' ', non '*')
    win_end = mature1_idx
    cnt = 0
    while win_end < n and cnt < A1A2_LEN:
        ch = ref_cols[win_end]
        if ch not in (".", " "):
            cnt += 1
        win_end += 1
    for nm in aligned:
        cols = resolve(nm)
        # full ungapped (leader+mature), drop gaps/blanks; '*' -> X
        full_seq = "".join("X" if c == "*" else c for c in cols if c not in (".", " "))
        full[nm] = full_seq
        mat_window = cols[mature1_idx:win_end]
        a = "".join("X" if c == "*" else c for c in mat_window if c not in (".", " "))
        a1a2[nm] = a
    return order, full, a1a2, mature1_idx


def main():
    align_full, align_a1a2 = {}, {}
    for loc in ("A", "B", "C"):
        order, full, a1a2, m1 = parse_alignment(f"{ALIGN_TMP}/{loc}_prot.txt")
        align_full.update(full)
        align_a1a2.update(a1a2)

    fasta = parse_fasta(HLA_FASTA)

    # ---- verification: reconstructed full == fasta, over all shared alleles ----
    shared = [a for a in align_full if a in fasta]
    ok = mism = 0
    examples = []
    for a in shared:
        if "X" in align_full[a] or "*" in fasta[a]:
            continue  # skip entries with unknown residues
        if align_full[a] == fasta[a]:
            ok += 1
        else:
            mism += 1
            if len(examples) < 8:
                examples.append(a)

    # ---- representative per 2-field (longest a1a2) ----
    rep = {}
    for a, s in align_a1a2.items():
        k = two_field(a)
        if not k:
            continue
        if k not in rep or len(s) > len(align_a1a2[rep[k]]):
            rep[k] = a

    # ---- dataset alleles ----
    alleles = [l.split("\t")[0].strip() for l in open(ALLELE_TSV).read().splitlines()[1:] if l.strip()]
    seqdict, rows, matched, miss = {}, [], 0, []
    for a in alleles:
        key = two_field(a[4:] if a.startswith("HLA-") else a)
        if a.startswith("HLA-") and key in rep:
            nm = rep[key]
            s = align_a1a2[nm]
            seqdict[a] = s
            rows.append((a, nm, "IMGT/HLA-align", str(len(s)), s))
            matched += 1
        else:
            miss.append(a)
            rows.append((a, "", "NO-ALIGN", "0", ""))

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(seqdict, open(f"{OUTDIR}/mhc_classI_a1a2.json", "w"), indent=0)
    with open(f"{OUTDIR}/mhc_classI_a1a2.tsv", "w") as fh:
        fh.write("allele\timgt_allele\tsource\ta1a2_length\ta1a2_sequence\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(f"{OUTDIR}/mhc_classI_a1a2.fasta", "w") as fh:
        for a, nm, src, ln, s in rows:
            if s:
                fh.write(f">{a} | {nm} | {src} | {ln}aa\n")
                for i in range(0, len(s), 80):
                    fh.write(s[i:i+80] + "\n")

    with open(f"{OUTDIR}/a1a2_alignment_verification.txt", "w") as fh:
        fh.write("IMGT/HLA alignment verification (A/B/C_prot.txt, full reconstruction vs hla_prot.fasta)\n")
        fh.write(f"shared alleles checked : {ok+mism}\n")
        fh.write(f"  exact match          : {ok}\n")
        fh.write(f"  mismatch             : {mism}\n")
        if examples:
            fh.write(f"  mismatch examples    : {', '.join(examples)}\n")
        fh.write("\nDataset alpha1+alpha2 extraction (mature residues 1-182):\n")
        fh.write(f"  dataset alleles      : {len(alleles)}\n")
        fh.write(f"  HLA matched via align: {matched}\n")
        fh.write(f"  not via alignment    : {len(miss)} -> {', '.join(miss)}\n")
        # length distribution of a1a2
        from collections import Counter
        lc = Counter(len(s) for s in seqdict.values())
        fh.write(f"  a1a2 length counts   : {dict(sorted(lc.items()))}\n")

    print(f"verification: {ok} match / {mism} mismatch (of {ok+mism} checked)")
    print(f"dataset: {matched} HLA via alignment, {len(miss)} not via alignment")
    print("a1a2 length counts:", dict(sorted(__import__('collections').Counter(len(s) for s in seqdict.values()).items())))
    if miss:
        print("not via alignment:", ", ".join(miss[:60]))


if __name__ == "__main__":
    main()
