#!/usr/bin/env python3
"""CDR3<->peptide interface PAE per complex from residue_stats_ranked_0.npz.

For each (ipTM-gated) complex, load the PAE matrix and average the block between the
CDR3 residues (from residue_frame_features) and the peptide residues, symmetrized:
  interface_pae = 0.5*(mean PAE[CDR3, peptide] + mean PAE[peptide, CDR3]).
Lower = the binding interface is more confidently positioned. Resumable.
"""
import argparse, csv, os
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)      # parent project root
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rff = pd.read_csv(f"{a.repo}/tcrpmhc_pose_binding/data/10x_cd8_A0201_6pep/residue_frame_features.csv.gz")
    idx = {}                                        # id -> (cdr3a idx, cdr3b idx, peptide idx)
    for cid, g in rff.groupby("id"):
        idx[cid] = (g[g.chain=="cdr3a"].idx.to_numpy(), g[g.chain=="cdr3b"].idx.to_numpy(),
                    g[g.chain=="peptide"].idx.to_numpy())
    seq = pd.read_csv(f"{a.repo}/data/10x_cd8_A0201_6pep/inputs/combined_sorted.csv")[["id"]]
    ip = pd.read_csv(f"{a.repo}/experiments/results/iptm_table.csv")
    gated = set(seq.merge(ip, on="id").query("tcr_pmhc_iptm>=0.5").id) & set(idx)

    done = set()
    if os.path.exists(a.out):
        done = {r["id"] for r in csv.DictReader(open(a.out))}
    todo = sorted(gated - done)
    sdir = f"{a.repo}/data/10x_cd8_A0201_6pep/structures"
    mode = "a" if done else "w"
    with open(a.out, mode, newline="") as fh:
        w = csv.writer(fh)
        if mode == "w": w.writerow(["id", "cdr3pep_pae"])
        n = 0
        for cid in todo:
            npz = f"{sdir}/{cid}/residue_stats_ranked_0.npz"
            if not os.path.exists(npz): continue
            try:
                z = np.load(npz); pae = z["pae"]; La, Lb, Lp = [int(x) for x in z["chain_lengths"][:3]]
                ia, ib, ip_ = idx[cid]
                cdr3 = np.concatenate([ia, La + ib]).astype(int)
                pep = (La + Lb + ip_).astype(int)
                cdr3 = cdr3[cdr3 < pae.shape[0]]; pep = pep[pep < pae.shape[0]]
                if len(cdr3) == 0 or len(pep) == 0: continue
                val = 0.5*(pae[np.ix_(cdr3, pep)].mean() + pae[np.ix_(pep, cdr3)].mean())
                w.writerow([cid, round(float(val), 4)]); n += 1
                if n % 500 == 0: fh.flush(); print(f"  {n}", flush=True)
            except Exception as e:
                print(f"  skip {cid}: {e}", flush=True)
            if a.limit and n >= a.limit: break
    print(f"wrote +{n} to {a.out}")

if __name__ == "__main__":
    main()
