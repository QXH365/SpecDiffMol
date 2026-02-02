# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolAlign, rdFMCS
from rdkit import RDLogger

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')

# =============================================================================
#  Helper Functions
# =============================================================================

def _prep_mol(mol, heavy_only:  bool):
    """Preprocess molecule (remove hydrogens)"""
    if mol is None: 
        return None
    m = Chem. Mol(mol)
    return Chem.RemoveHs(m) if heavy_only else m


def _coords(mol):
    """Get molecule coordinates"""
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    return np.asarray(pos, dtype=float)


def get_mol_props(smiles):
    """Get molecular formula and heavy atom count"""
    if not smiles:
        return None, -1
    
    smiles = smiles.replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None, -1
    
    formula = rdMolDescriptors.CalcMolFormula(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    
    return formula, heavy_atoms


def standardize_smiles(smiles:  str) -> str:
    """Standardize SMILES"""
    if not smiles: 
        return ""
    s = smiles.replace(" ", "")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ""
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def mol_to_coords_str(mol):
    """Extract molecule coordinates for CSV display"""
    if mol is None or mol.GetNumConformers() == 0:
        return ""
    
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    
    return "; ".join([f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}" for row in coords])


# =============================================================================
#  Core RMSD Calculation Function
# =============================================================================

def mcs_align_plus_hungarian(
    ref_mol, gen_mol,
    heavy_only=True,
    mcs_timeout=10,
    max_matches=30,
    max_pair_trials=400,
    element_mismatch_penalty=1e6,
):
    """
    Atom counts must match, otherwise return atom_count_mismatch (discard RMSD directly).
    Returns:
      rmsd_total (All atoms, MCS+Hungarian aligned),
      rmsd_mcs   (Only MCS part aligned using AlignMol),
      mcs_atoms,
      n_extra,
      status
    """
    if ref_mol is None or gen_mol is None:
        return None, None, 0, 0, "no_mol"

    ref = _prep_mol(ref_mol, heavy_only)
    gen = _prep_mol(gen_mol, heavy_only)

    if ref is None or gen is None:
        return None, None, 0, 0, "prep_mol_failed"

    if ref.GetNumConformers() == 0 or gen.GetNumConformers() == 0:
        return None, None, 0, 0, "no_conformer"

    n_ref = ref.GetNumAtoms()
    n_gen = gen.GetNumAtoms()
    if n_ref != n_gen:
        return None, None, 0, 0, "atom_count_mismatch"

    try:
        mcs = rdFMCS.FindMCS(
            [ref, gen],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=mcs_timeout,
        )
    except Exception: 
        return None, None, 0, 0, "mcs_exception"

    if not mcs or mcs.numAtoms <= 0 or not mcs.smartsString:
        return None, None, 0, 0, "mcs_not_found"

    q = Chem.MolFromSmarts(mcs.smartsString)
    if q is None:
        return None, None, 0, 0, "mcs_smarts_bad"

    ref_matches = ref.GetSubstructMatches(q, uniquify=False, maxMatches=max_matches)
    gen_matches = gen.GetSubstructMatches(q, uniquify=False, maxMatches=max_matches)
    if not ref_matches or not gen_matches:
        return None, None, int(mcs.numAtoms), 0, "mcs_match_empty"

    ref_symbols = [a.GetSymbol() for a in ref. GetAtoms()]
    gen_symbols = [a.GetSymbol() for a in gen.GetAtoms()]
    ref_all = np.arange(n_ref)
    best = {
        "rmsd_total": None,
        "rmsd_mcs": None,
        "mcs_atoms": int(mcs.numAtoms),
        "n_extra": 0,
        "status": "failed",
    }

    trials = 0
    for rm in ref_matches:
        rm_set = set(rm)
        for gm in gen_matches: 
            trials += 1
            if trials > max_pair_trials:
                break

            # Align first: Copy gen to avoid modifying original
            gen_copy = Chem.Mol(gen)
            atom_map = list(zip(gm, rm))  # (gen_idx, ref_idx)

            try:
                rmsd_mcs = float(rdMolAlign.AlignMol(gen_copy, ref, atomMap=atom_map))
            except Exception:
                continue

            gm_set = set(gm)

            # Unmatched atoms set
            u_ref = [i for i in ref_all if i not in rm_set]
            u_gen = [i for i in ref_all if i not in gm_set]  # n_ref == n_gen, use ref_all works

            # Number of matches must be equal
            if len(u_ref) != len(u_gen):
                continue

            # Hungarian: Based on aligned gen_copy coordinates
            try:
                ref_xyz = _coords(ref)
                gen_xyz = _coords(gen_copy)
            except Exception:
                continue

            n_u = len(u_ref)
            extra_pairs = []
            if n_u > 0:
                cost = np.zeros((n_u, n_u), dtype=float)
                for ii, gi in enumerate(u_gen):
                    for jj, rj in enumerate(u_ref):
                        d = np.linalg.norm(gen_xyz[gi] - ref_xyz[rj])
                        if gen_symbols[gi] != ref_symbols[rj]:
                            d += element_mismatch_penalty
                        cost[ii, jj] = d

                try: 
                    row_ind, col_ind = linear_sum_assignment(cost)
                except Exception:
                    continue

                # Check element consistency
                ok = True
                for ii, jj in zip(row_ind, col_ind):
                    gi = u_gen[ii]
                    rj = u_ref[jj]
                    if gen_symbols[gi] != ref_symbols[rj]:
                        ok = False
                        break
                    extra_pairs.append((gi, rj))
                if not ok:
                    continue

            # All pairs: MCS pairs + extra pairs
            all_pairs = atom_map + extra_pairs

            # Calculate total RMSD
            sq = 0.0
            for gi, rj in all_pairs: 
                diff = gen_xyz[gi] - ref_xyz[rj]
                sq += float(diff. dot(diff))
            rmsd_total = float(np.sqrt(sq / len(all_pairs)))
            
            if best["rmsd_total"] is None or rmsd_total < best["rmsd_total"]:
                best.update({
                    "rmsd_total": rmsd_total,
                    "rmsd_mcs":  rmsd_mcs,
                    "mcs_atoms":mcs.numAtoms,
                    "n_extra": len(extra_pairs),
                    "status": "ok",
                })

        if trials > max_pair_trials:
            break

    if best["status"] != "ok":
        return None, None, int(mcs.numAtoms), 0, "align_or_hungarian_failed"

    return best["rmsd_total"], best["rmsd_mcs"], best["mcs_atoms"], best["n_extra"], "ok"


# =============================================================================
#  Main Program
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluation Script: MCS + Hungarian Method')
    parser.add_argument('--result_pkl', type=str, required=True,
                       help='Inference pickle file')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV filename')
    parser.add_argument('--use_all_atoms', action='store_true',
                       help='Use all atoms (including Hydrogen) for RMSD (Default: Heavy atoms only)')
    parser.add_argument('--mcs_timeout', type=int, default=10,
                       help='MCS search timeout (seconds)')
    parser.add_argument('--max_matches', type=int, default=30,
                       help='MCS max matches')
    parser.add_argument('--max_pair_trials', type=int, default=400,
                       help='MCS match pair max trials')
    args = parser.parse_args()

    print(f"Loading inference results: {args.result_pkl} ...")
    if not os.path.exists(args.result_pkl):
        print("Error: File not found")
        return

    with open(args.result_pkl, 'rb') as f:
        data_list = pickle.load(f)
    
    heavy_only = not args.use_all_atoms
    
    print(f"Loaded {len(data_list)} samples")
    print(f"RMSD Calculation Mode: {'All Atoms (with H)' if args.use_all_atoms else 'Heavy Atoms Only'}")
    print(f"Evaluation Strategy: MCS + Hungarian")
    
    csv_rows = []
    
    # Stats counters
    stats = {
        'total_samples': len(data_list),
        'total_confs': 0,
        'valid_rmsd_count': 0,
        'atom_count_mismatch':  0,
        'failed_count': 0,
    }
    
    all_valid_rmsds = []
    all_mcs_coverage = []
    all_extra_atoms = []

    # Iterate over test samples
    for i, item in tqdm(enumerate(data_list), total=len(data_list), desc="Processing"):
        true_smiles = item. get('true_smiles', '') or ""
        pred_smiles = item.get('pred_smiles', '') or ""
        true_mol = item.get('true_mol', None)
        gen_mols = item.get('gen_mols', None) or []
        
        # 1. Basic property check
        smi_correct = (true_smiles. replace(' ', '') == pred_smiles.replace(' ', '')) \
                      if (true_smiles and pred_smiles) else False
        
        true_formula, true_heavy = get_mol_props(true_smiles)
        pred_formula, pred_heavy = get_mol_props(pred_smiles)
        
        formula_match = (true_formula == pred_formula) if true_formula else False
        heavy_atom_match = (true_heavy == pred_heavy) if (true_heavy != -1) else False
        
        # Standardize SMILES
        true_std = standardize_smiles(true_smiles)
        pred_std = standardize_smiles(pred_smiles)
        
        # Prepare true molecule coords string
        pos_true_str = mol_to_coords_str(true_mol)

        # 2. Process generated conformers
        if not gen_mols:
            csv_rows.append({
                'Index': f"{i}_Err",
                'True SMILES': true_std,
                'Predicted SMILES': pred_std,
                'SMILES Correct': smi_correct,
                'Formula Match': formula_match,
                'Heavy Atom Match': heavy_atom_match,
                'RMSD (Å)': "",
                'Note': 'No generated mols',
                'Status': 'no_gen_mols'
            })
            continue

        # Denominator: True atom count
        true_for_cov = _prep_mol(true_mol, heavy_only) if true_mol is not None else None
        n_true_atoms = true_for_cov.GetNumAtoms() if true_for_cov is not None else 0

        best_rmsd_total = None
        best_rmsd_mcs = None
        best_conf_id = None
        best_mcs_atoms = 0
        best_extra = 0
        status = "no_true_mol" if true_mol is None else "failed"

        if true_mol is not None:
            # Pick the one with smallest RMSD_total (assuming atom count matches)
            for k, gm in enumerate(gen_mols):
                stats['total_confs'] += 1
                
                rmsd_total, rmsd_mcs, mcs_atoms, n_extra, st = mcs_align_plus_hungarian(
                    true_mol, gm,
                    heavy_only=heavy_only,
                    mcs_timeout=args.mcs_timeout,
                    max_matches=args.max_matches,
                    max_pair_trials=args.max_pair_trials,
                )
                
                # Atom count mismatch
                if st == "atom_count_mismatch":
                    stats['atom_count_mismatch'] += 1
                    status = "atom_count_mismatch"
                    continue
                
                if st != "ok" or rmsd_total is None:
                    # Continue to next conformer
                    status = st
                    continue

                if best_rmsd_total is None or rmsd_total < best_rmsd_total:
                    best_rmsd_total = rmsd_total
                    best_rmsd_mcs = rmsd_mcs
                    best_conf_id = k
                    best_mcs_atoms = int(mcs_atoms)
                    best_extra = int(n_extra)
                    status = "ok"

        # Calculate MCS coverage
        mcs_pct = ""
        if status == "ok" and n_true_atoms > 0:
            mcs_pct = f"{(best_mcs_atoms / n_true_atoms * 100.0):.2f}"
            all_mcs_coverage.append(best_mcs_atoms / n_true_atoms * 100.0)
            all_extra_atoms.append(best_extra)

        # Update stats
        if status == "ok":
            stats['valid_rmsd_count'] += 1
            all_valid_rmsds.append(best_rmsd_total)
        else:
            stats['failed_count'] += 1

        # Get best conformer coords
        pos_pred_str = ""
        if best_conf_id is not None and gen_mols:
            pos_pred_str = mol_to_coords_str(gen_mols[best_conf_id])

        # Add to CSV row
        csv_rows.append({
            'Index':  i,
            'True SMILES': true_std,
            'Predicted SMILES': pred_std,
            'SMILES Correct': smi_correct,
            'Formula Match': formula_match,
            'Heavy Atom Match': heavy_atom_match,
            'Total Atoms': n_true_atoms if n_true_atoms > 0 else "",
            'MCS Atoms':  best_mcs_atoms if status == "ok" else "",
            'MCS Coverage (%)': mcs_pct,
            'Extra Atoms (Hungarian)': best_extra if status == "ok" else "",
            'RMSD Total (Å)': f"{best_rmsd_total:.4f}" if best_rmsd_total is not None else "",
            'RMSD MCS Only (Å)': f"{best_rmsd_mcs:.4f}" if best_rmsd_mcs is not None else "",
            'RMSD < 0.5':  (best_rmsd_total < 0.5) if (best_rmsd_total is not None) else False,
            'RMSD < 1.0':  (best_rmsd_total < 1.0) if (best_rmsd_total is not None) else False,
            'RMSD < 2.0': (best_rmsd_total < 2.0) if (best_rmsd_total is not None) else False,
            'Best Conf ID': best_conf_id if best_conf_id is not None else "",
            'True Conformation': pos_true_str,
            'Predicted Conformation': pos_pred_str,
            'Status': status,
            'Heavy Only': heavy_only,
        })

    # --- Save CSV ---
    df = pd.DataFrame(csv_rows)
    
    print(f"\nWriting to file: {args.output_csv}")
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    
    # --- Print Stats Summary ---
    print("\n" + "="*70)
    print("📊 Stats Summary (MCS + Hungarian Method)")
    print("="*70)
    
    print(f"Calc Mode: {'All Atoms (with H)' if args.use_all_atoms else 'Heavy Atoms Only'}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Total Confs: {stats['total_confs']}")
    print("-" * 70)
    print(f"Valid RMSD: {stats['valid_rmsd_count']} ({stats['valid_rmsd_count']/stats['total_samples']*100:.2f}% samples)")
    print(f"Atom Count Mismatch: {stats['atom_count_mismatch']}")
    print(f"Failed Calc: {stats['failed_count']}")
    
    if all_mcs_coverage:
        print(f"\nMCS Coverage Stats (n={len(all_mcs_coverage)}):")
        print(f"  - Mean Coverage: {np.mean(all_mcs_coverage):.2f}%")
        print(f"  - Median Coverage: {np.median(all_mcs_coverage):.2f}%")
        print(f"  - Min Coverage:  {min(all_mcs_coverage):.2f}%")
        print(f"  - Max Coverage: {max(all_mcs_coverage):.2f}%")
        
        print(f"\nHungarian Extra Atoms Stats:")
        print(f"  - Mean: {np.mean(all_extra_atoms):.2f}")
        print(f"  - Median: {np.median(all_extra_atoms):.2f}")
        print(f"  - Max: {max(all_extra_atoms)}")
    
    if all_valid_rmsds: 
        mean_r = np.mean(all_valid_rmsds)
        median_r = np.median(all_valid_rmsds)
        pass_05 = np.mean([r < 0.5 for r in all_valid_rmsds]) * 100
        pass_10 = np.mean([r < 1.0 for r in all_valid_rmsds]) * 100
        pass_20 = np.mean([r < 2.0 for r in all_valid_rmsds]) * 100
        
        print("\n" + "-" * 70)
        print("[RMSD Stats]")
        print("-" * 70)
        print(f"Mean RMSD:            {mean_r:.4f} Å")
        print(f"Median RMSD:           {median_r:.4f} Å")
        print(f"RMSD < 0.5 Å:        {pass_05:.2f}%")
        print(f"RMSD < 1.0 Å:        {pass_10:.2f}%")
        print(f"RMSD < 2.0 Å:        {pass_20:.2f}%")
        print(f"Min RMSD:           {min(all_valid_rmsds):.4f} Å")
        print(f"Max RMSD:           {max(all_valid_rmsds):.4f} Å")
    else:
        print("\n⚠️ No Valid RMSD Data Calculated")
    
    print("="*70)

if __name__ == "__main__":
    main()