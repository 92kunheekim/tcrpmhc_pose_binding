#!/usr/bin/env python3
"""Add alpha1+alpha2 for the non-human class I alleles (Patr, Mamu, SLA, BoLA,
Eqca, DLA, Gaga-BF, Gogo) that are absent from the IMGT/HLA alignment.

No IMGT alignment ships for IPD-MHC locally, so the groove is extracted by
local alignment (BLOSUM62) of each allele's full IPD-MHC protein against the
VERIFIED HLA reference alpha1+alpha2 (HLA-A*02:01, 182 aa from the alignment
step). The homologous target span is taken as alpha1+alpha2 and labelled
'IPD-MHC-homology' with the % identity to the HLA reference for QC.

Merges with the alignment-verified HLA groove sequences into the final
combined dictionary (all alleles with an available sequence).
"""
import json, re, os
from Bio import Align
from Bio.Align import substitution_matrices

MHC_FASTA = "/sessions/eager-sharp-pasteur/mnt/IMGTMHC/MHC_prot.fasta"
PROC = ("/sessions/eager-sharp-pasteur/mnt/tcrpmhc_model_structure_reliability/"
        "tcrpmhc_pose_binding/data/pmhc_classI_pretraining/processed")
ALLELE_TSV = f"{PROC}/iedb_mhc_classI_alleles.tsv"


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


def main():
    hla_a1a2 = json.load(open(f"{PROC}/mhc_classI_a1a2.json"))
    ref = hla_a1a2["HLA-A*02:01"]                       # verified 182-aa groove

    mhc = parse_fasta(MHC_FASTA)
    # 2-field representative (longest seq) for non-human
    rep = {}
    for nm, s in mhc.items():
        k = two_field(nm)
        if not k:
            continue
        if k not in rep or len(s) > len(mhc[rep[k]]):
            rep[k] = nm

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.mode = "local"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    def extract(fullseq):
        aln = aligner.align(fullseq, ref)[0]
        tblocks = aln.aligned[0]
        rblocks = aln.aligned[1]
        t0 = tblocks[0][0]
        t1 = tblocks[-1][1]
        span = fullseq[t0:t1]
        # identity over aligned columns
        ident = tot = 0
        for (ts, te), (rs, re_) in zip(tblocks, rblocks):
            for a, b in zip(fullseq[ts:te], ref[rs:re_]):
                tot += 1
                ident += (a == b)
        return span, (100.0 * ident / tot if tot else 0.0)

    alleles = [l.split("\t")[0].strip() for l in open(ALLELE_TSV).read().splitlines()[1:] if l.strip()]

    combined = dict(hla_a1a2)        # start with verified HLA
    rows, miss, nonhuman = [], [], 0
    qc = []
    for a in alleles:
        if a in hla_a1a2:
            continue  # handled (HLA)
        key = two_field(a)
        nm = rep.get(key)
        if nm:
            span, ident = extract(mhc[nm])
            combined[a] = span
            rows.append((a, nm, "IPD-MHC-homology", str(len(span)), f"{ident:.1f}", span))
            nonhuman += 1
            qc.append((a, len(span), ident))
        else:
            miss.append(a)

    # write combined dict
    json.dump(combined, open(f"{PROC}/mhc_classI_a1a2.json", "w"), indent=0)

    # rebuild combined tsv (HLA verified + non-human homology)
    # reload HLA tsv rows
    hla_rows = []
    for l in open(f"{PROC}/mhc_classI_a1a2.tsv").read().splitlines()[1:]:
        p = l.split("\t")
        if len(p) >= 5 and p[0] in hla_a1a2:
            hla_rows.append((p[0], p[1], p[2], p[3], "100.0", p[4]))
    with open(f"{PROC}/mhc_classI_a1a2.tsv", "w") as fh:
        fh.write("allele\timgt_allele\tsource\ta1a2_length\tpct_id_to_HLA_ref\ta1a2_sequence\n")
        for r in sorted(hla_rows) + sorted(rows):
            fh.write("\t".join(r) + "\n")
    with open(f"{PROC}/mhc_classI_a1a2.fasta", "w") as fh:
        for a, s in combined.items():
            fh.write(f">{a} | {len(s)}aa\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i+80] + "\n")

    print(f"HLA (alignment-verified): {len(hla_a1a2)}")
    print(f"non-human (homology)    : {nonhuman}")
    print(f"total in dict           : {len(combined)}")
    print(f"still missing (no seq)  : {len(miss)} -> {', '.join(miss)}")
    print("non-human QC (allele, a1a2_len, %id_to_HLA):")
    for a, l, i in sorted(qc, key=lambda x: x[2]):
        print(f"  {a:16s} len={l:3d}  id={i:4.1f}%")


if __name__ == "__main__":
    main()
