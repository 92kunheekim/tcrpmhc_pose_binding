#!/usr/bin/env python3
"""Map every class I allele in the IEDB dataset to its representative protein
sequence from IMGT/HLA (hla_prot.fasta) and IPD-MHC (MHC_prot.fasta).

Matching is at 2-field (4-digit) resolution: a dataset allele like HLA-A*02:01
is matched to all IMGT entries whose locus*field1:field2 == A*02:01, and the
longest protein sequence in that group is taken as the representative (3rd/4th
fields are synonymous/non-coding, so the protein is identical modulo partial
sequencing).
"""
import json, re, sys, os

HLA_FASTA = "/sessions/eager-sharp-pasteur/mnt/IMGTHLA/hla_prot.fasta"
MHC_FASTA = "/sessions/eager-sharp-pasteur/mnt/IMGTMHC/MHC_prot.fasta"
ALLELE_TSV = ("/sessions/eager-sharp-pasteur/mnt/tcrpmhc_model_structure_reliability/"
              "tcrpmhc_pose_binding/data/pmhc_classI_pretraining/processed/"
              "iedb_mhc_classI_alleles.tsv")
OUTDIR = ("/sessions/eager-sharp-pasteur/mnt/tcrpmhc_model_structure_reliability/"
          "tcrpmhc_pose_binding/data/pmhc_classI_pretraining/processed")


def two_field(name):
    """Return locus*field1:field2 key, dropping expression suffix letters.
    e.g. A*01:01:01:01 -> A*01:01 ; Patr-A*09:01:01 -> Patr-A*09:01"""
    if "*" not in name:
        return None
    locus, rest = name.split("*", 1)
    fields = rest.split(":")
    if len(fields) < 2:
        return None
    f1 = fields[0]
    f2 = re.sub(r"[A-Za-z]+$", "", fields[1])  # strip N/L/S/Q suffix for the key
    return f"{locus}*{f1}:{f2}"


def parse_fasta(path, source):
    """Yield (allele_name, sequence, source). Header: '>ID allele len bp'."""
    entries = []
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(seq), source))
                parts = line[1:].split()
                name = parts[1] if len(parts) > 1 else parts[0]
                seq = []
            else:
                seq.append(line.strip())
        if name is not None:
            entries.append((name, "".join(seq), source))
    return entries


def build_index(entries):
    """2-field key -> (best_full_name, best_seq, source) keeping longest seq."""
    idx = {}
    for name, seq, source in entries:
        k = two_field(name)
        if not k:
            continue
        cur = idx.get(k)
        if cur is None or len(seq) > len(cur[1]):
            idx[k] = (name, seq, source)
    return idx


def main():
    hla = parse_fasta(HLA_FASTA, "IMGT/HLA")
    mhc = parse_fasta(MHC_FASTA, "IPD-MHC")
    idx = build_index(hla)
    # IPD-MHC second so HLA wins on any collision (shouldn't collide: prefixes differ)
    for k, v in build_index(mhc).items():
        idx.setdefault(k, v)

    # dataset alleles
    alleles = []
    with open(ALLELE_TSV) as fh:
        next(fh)
        for line in fh:
            a = line.split("\t")[0].strip()
            if a:
                alleles.append(a)

    seq_dict = {}
    rows = []
    matched = unmatched = 0
    miss = []
    for a in alleles:
        key = a[4:] if a.startswith("HLA-") else a   # strip HLA- for HLA names
        key = two_field(key)
        hit = idx.get(key)
        if hit:
            full, seq, source = hit
            seq_dict[a] = seq
            rows.append((a, full, source, str(len(seq)), seq))
            matched += 1
        else:
            miss.append(a)
            rows.append((a, "", "UNMATCHED", "0", ""))
            unmatched += 1

    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/mhc_classI_sequences.json", "w") as fh:
        json.dump(seq_dict, fh, indent=0)
    with open(f"{OUTDIR}/mhc_classI_sequences.tsv", "w") as fh:
        fh.write("allele\timgt_allele\tsource\tlength\tsequence\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(f"{OUTDIR}/mhc_classI_sequences.fasta", "w") as fh:
        for a, full, source, length, seq in rows:
            if seq:
                fh.write(f">{a} | {full} | {source} | {length}aa\n")
                for i in range(0, len(seq), 80):
                    fh.write(seq[i:i+80] + "\n")

    print(f"dataset alleles : {len(alleles)}")
    print(f"matched         : {matched}")
    print(f"unmatched       : {unmatched}")
    if miss:
        print("UNMATCHED:", ", ".join(miss))


if __name__ == "__main__":
    main()
