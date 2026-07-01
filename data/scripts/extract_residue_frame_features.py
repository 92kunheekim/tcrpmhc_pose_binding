#!/usr/bin/env python3
"""Per-residue groove-frame features for peptide + CDR3 residues, from RELAXED structures.

For each complex (relaxed AlphaFold-multimer PDB), build the pMHC groove frame (same fixed,
TCR-independent convention as descriptors/frame_pose.py, Supplementary Note 2.1) and express
each peptide and CDR3 residue as:
  along, across, height   Cα position in the groove frame (Å)
  sc_reach, dx, dy, dz     side-chain functional-group direction: reach (Å) + unit dir in frame
  plddt                    per-residue confidence (relaxed PDB B-factor)
  no_sc                    1 for glycine / no side chain (reach=0)

Side chains come from the RELAXED structure (better rotamers). Frame axes: x=along (peptide
N->C), z=height (toward peptide core / solvent-TCR side), y=across (right-handed). Positions
are RAW Å here; scale by trans_scale downstream for consistency with the body-pose features.

Usage:
  python extract_residue_frame_features.py --struct_dir <structures/> --out residue_frame_features.csv
Resumable: skips ids already present in --out.
"""
import argparse, csv, os, re, glob
import numpy as np
from Bio.PDB import PDBParser

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# functional-group atoms per residue (see chemistry classes); centroid = the direction target
FG = {"PHE":["CG","CD1","CD2","CE1","CE2","CZ"], "TYR":["CG","CD1","CD2","CE1","CE2","CZ"],
      "TRP":["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"], "HIS":["CG","ND1","CD2","CE1","NE2"],
      "LYS":["NZ"], "ARG":["CZ","NH1","NH2","NE"], "ASP":["OD1","OD2"], "GLU":["OE1","OE2"],
      "ASN":["OD1","ND2"], "GLN":["OE1","NE2"], "SER":["OG"], "THR":["OG1"], "CYS":["SG"], "MET":["SD"],
      "ALA":["CB"], "VAL":["CG1","CG2"], "LEU":["CD1","CD2"], "ILE":["CD1","CG2"], "PRO":["CB","CG","CD"]}
_BB = {"N", "CA", "C", "O", "OXT"}
_PAT = re.compile(r"C[A-Z]{4,28}?[FW]G[A-Z]G")


def _unit(v):
    n = np.linalg.norm(v); return v/n if n > 1e-9 else v


def build_frame(mhc_ca, pep_ca, pep_core):
    """pMHC groove frame: R_rows (3x3, rows = along/across/height axes) and origin (platform centroid).
    coord_in_frame = R_rows @ (p - origin). Matches frame_pose.calc_mhc_frame conventions."""
    cent = mhc_ca.mean(0)
    _, _, vh = np.linalg.svd(mhc_ca - cent, full_matrices=False)
    x = _unit(vh[0]); z = _unit(vh[2])
    if np.dot(x, _unit(pep_ca[-1] - pep_ca[0])) < 0: x = -x        # long axis -> peptide N->C
    if np.dot(z, pep_core.mean(0) - cent) < 0: z = -z             # normal -> peptide-core/solvent side
    y = _unit(np.cross(z, x)); z = _unit(np.cross(x, y))          # orthonormalize, right-handed
    if np.dot(z, pep_core.mean(0) - cent) < 0: z = -z; y = -y
    return np.vstack([x, y, z]), cent


def fg_centroid(residue):
    """Functional-group centroid; fallback to all side-chain heavy atoms; None for Gly/no side chain."""
    atoms = {a.name: a.coord for a in residue if a.element != "H"}
    names = FG.get(residue.resname)
    if names:
        pts = [atoms[n] for n in names if n in atoms]
        if pts: return np.mean(pts, 0)
    sc = [c for n, c in atoms.items() if n not in _BB]           # any side-chain heavy atom
    return np.mean(sc, 0) if sc else None


def residue_feats(res, R, cent):
    ca = res["CA"].coord
    along, across, height = R @ (ca - cent)
    fg = fg_centroid(res)
    if fg is None:
        reach, d, no_sc = 0.0, np.zeros(3), 1
    else:
        v = fg - ca; reach = float(np.linalg.norm(v)); d = R @ _unit(v); no_sc = 0
    return dict(aa=AA3.get(res.resname, "X"), along=float(along), across=float(across), height=float(height),
                sc_reach=reach, dx=float(d[0]), dy=float(d[1]), dz=float(d[2]),
                plddt=float(res["CA"].get_bfactor()), no_sc=no_sc)


def std_residues(model, chain):
    return [r for r in model[chain] if r.id[0] == " " and "CA" in r] if chain in model else []


def cdr3_span(residues):
    seq = "".join(AA3.get(r.resname, "X") for r in residues)
    h = list(_PAT.finditer(seq))
    if not h: return None
    m = h[-1]; return m.start()+1, m.end()-4                     # core: drop leading C + trailing FGXG


def process_one(pdb_path, cid, mhc_chain="D", pep_chain="C", tcra_chain="A", tcrb_chain="B"):
    model = PDBParser(QUIET=True).get_structure(cid, pdb_path)[0]
    resD, resC = std_residues(model, mhc_chain), std_residues(model, pep_chain)
    if len(resC) < 2 or len(resD) < 3: return []
    mhc_ca = np.array([r["CA"].coord for r in resD if 1 <= int(r.id[1]) <= 150])
    if len(mhc_ca) < 3: mhc_ca = np.array([r["CA"].coord for r in resD[:150]])
    pep_ca = np.array([r["CA"].coord for r in resC])
    core = pep_ca[2:-2] if len(pep_ca) >= 9 else pep_ca
    R, cent = build_frame(mhc_ca, pep_ca, core)
    rows = []
    for k, r in enumerate(resC):
        rows.append(dict(id=cid, chain="peptide", idx=k, **residue_feats(r, R, cent)))
    for tag, chain in (("cdr3a", tcra_chain), ("cdr3b", tcrb_chain)):
        res = std_residues(model, chain); span = cdr3_span(res)
        if not span: continue
        for k in range(*span):
            rows.append(dict(id=cid, chain=tag, idx=k, **residue_feats(res[k], R, cent)))
    return rows


COLS = ["id", "chain", "idx", "aa", "along", "across", "height", "sc_reach", "dx", "dy", "dz", "plddt", "no_sc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct_dir", required=True, help="dir of <id>/ranked_0_relaxed.pdb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb_name", default="ranked_0_relaxed.pdb")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
    dirs = sorted(d for d in glob.glob(os.path.join(args.struct_dir, "*")) if os.path.isdir(d))
    new = "w" if not done else "a"
    with open(args.out, new, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new == "w": w.writeheader()
        n = 0
        for d in dirs:
            cid = os.path.basename(d)
            if cid in done: continue
            pdb = os.path.join(d, args.pdb_name)
            if not os.path.exists(pdb): continue
            try:
                for row in process_one(pdb, cid): w.writerow(row)
                n += 1
                if n % 200 == 0: fh.flush(); print(f"  {n} done", flush=True)
            except Exception as e:
                print(f"  skip {cid}: {e}", flush=True)
            if args.limit and n >= args.limit: break
    print(f"wrote/updated {args.out} (+{n} complexes)")


if __name__ == "__main__":
    main()
