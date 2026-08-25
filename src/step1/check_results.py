import csv
import sys
import argparse
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

# --- 1. Config & Globals ---

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- 2. Functional Group Definitions ---
FG_SMARTS = {
    "Alkane": "[CX4]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#C",
    "Arene": "[$([cX3](:*):*),$([cX2+](:*):*)]",
    "Alcohol": "[#6][OX2H]",
    "Ether": "[OD2]([#6])[#6]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Carboxylicacid": "[CX3](=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "haloalkane": "[#6][F,Cl,Br,I]",
    "Alkylhalide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Amine": "[NX3;$(NC=O)]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Sulfide": "[#16X2H0]",
    "Thiol": "[#16X2H]"
}

FG_PATTERNS = {}
for name, smarts in FG_SMARTS.items():
    mol_pat = Chem.MolFromSmarts(smarts)
    if mol_pat:
        FG_PATTERNS[name] = mol_pat

# --- 3. Core Helper Functions ---

def get_mol(smiles_string: str):
    """Clean and return RDKit molecule object"""
    try:
        if not smiles_string:
            return None
        cleaned_smiles = "".join(smiles_string.strip().split())
        mol = Chem.MolFromSmiles(cleaned_smiles)
        return mol
    except:
        return None

def get_canonical_smiles(mol):
    """Return canonical SMILES"""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def get_murcko_scaffold_smiles(mol):
    """Get Murcko Scaffold SMILES (for exact matching)"""
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True) # Return self if acyclic
        return Chem.MolToSmiles(scaffold, canonical=True)
    except:
        return None

def get_murcko_scaffold_mol(mol):
    """Get Murcko Scaffold Mol object (for fingerprint calculation)"""
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return scaffold 
    except:
        return None

def identify_func_groups(mol):
    """Identify functional groups in molecule, return semicolon-separated string"""
    if mol is None:
        return "Invalid_Mol"
    found_fgs = []
    for name, pattern in FG_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            found_fgs.append(name)
    if not found_fgs:
        return "None"
    return ";".join(sorted(found_fgs))

# --- Metric Calculation Functions ---

def calculate_mces(mol1, mol2):
    """
    Calculate real MCES distance (based on Graph Edit Distance of edges)
    
    Args:
        mol1: Target Molecule
        mol2: Generated Molecule
        
    Returns:
        ratio: Bond Recall (MCS Bonds / Target Bonds)
        distance: MCES Distance (|E1| + |E2| - 2*|Emcs|)
    """
    try:
        # 1. Handle empty target
        if mol1 is None:
            return 0.0, 0
        
        num_bonds1 = mol1.GetNumBonds()
        
        # 2. Handle empty generated molecule
        if mol2 is None:
            # Distance = Target bonds (need to cut all edges)
            return 0.0, num_bonds1 

        # 3. Special case: Target is bondless (e.g. Methane CH4)
        if num_bonds1 == 0:
            num_bonds2 = mol2.GetNumBonds()
            if num_bonds2 == 0:
                # Both are bondless, approximate match
                return 1.0, 0
            return 0.0, num_bonds2

        # 4. Find MCS
        # Critical: BondCompare uses CompareOrder for strict bond order matching
        mcs_result = rdFMCS.FindMCS([mol1, mol2], 
                                    timeout=2, 
                                    bondCompare=rdFMCS.BondCompare.CompareOrder, 
                                    atomCompare=rdFMCS.AtomCompare.CompareElements)
        
        num_common_edges = mcs_result.numBonds
        num_bonds2 = mol2.GetNumBonds()
        
        # 5. Calculate MCES Distance (Graph Edit Distance)
        # Formula: D = |E1| + |E2| - 2 * |Emcs|
        mces_distance = num_bonds1 + num_bonds2 - 2 * num_common_edges
        
        # Calculate Ratio (Bond Recall)
        mces_ratio = num_common_edges / num_bonds1
        
        return mces_ratio, mces_distance

    except Exception as e:
        # Conservative estimate on error
        num_bonds1 = mol1.GetNumBonds() if mol1 else 0
        return 0.0, num_bonds1

def calculate_fg_similarity(mol1, mol2):
    """Calculate Functional Group Jaccard Similarity"""
    if mol1 is None or mol2 is None:
        return 0.0
    
    def get_fg_set(m):
        found = set()
        for name, pattern in FG_PATTERNS.items():
            if m.HasSubstructMatch(pattern):
                found.add(name)
        return found

    fg1 = get_fg_set(mol1)
    fg2 = get_fg_set(mol2)
    
    intersection = len(fg1 & fg2)
    union = len(fg1 | fg2)
    
    if union == 0:
        return 1.0 
    return intersection / union

def calculate_tanimoto_from_mols(mol1, mol2):
    """Calculate Morgan Fingerprint Tanimoto Similarity"""
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def calculate_scaffold_fingerprint_similarity(mol1, mol2):
    """Calculate Scaffold Tanimoto Similarity"""
    if mol1 is None or mol2 is None:
        return 0.0
    scaf1 = get_murcko_scaffold_mol(mol1)
    scaf2 = get_murcko_scaffold_mol(mol2)
    if scaf1 is None or scaf2 is None:
        return 0.0
    # If both are acyclic (no scaffold atoms), consider scaffolds identical
    if scaf1.GetNumAtoms() == 0 and scaf2.GetNumAtoms() == 0:
        return 1.0
    return calculate_tanimoto_from_mols(scaf1, scaf2)

# --- 4. Main Processing Logic ---

def process_files(label_path, top10_path, top1_path, output_csv_path, report_path, num_preds_per_sample=10):
    print("Loading files...")
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = [line.strip() for line in f.readlines()]
        with open(top10_path, 'r', encoding='utf-8') as f:
            top10_lines = [line.strip() for line in f.readlines()]
        with open(top1_path, 'r', encoding='utf-8') as f:
            top1_lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return

    # Verify line counts
    num_samples = len(label_lines)
    if len(top1_lines) != num_samples:
        print(f"Error: Top-1 file line count mismatch! Expected {num_samples}, Got {len(top1_lines)}")
        return
    if len(top10_lines) != num_samples * num_preds_per_sample:
        # Try to adapt if mismatch (warn user)
        if len(top10_lines) > num_samples * num_preds_per_sample:
             print(f"Warning: Top-10 file has more lines than expected. Truncating.")
        else:
             print(f"Error: Top-10 file line count mismatch! Expected {num_samples * num_preds_per_sample}, Got {len(top10_lines)}")
             return

    print(f"Validation passed: {num_samples} samples. Calculating Top-1, Top-5, Top-10 metrics...")

    results = []
    
    # Stats accumulators (for averages)
    keys = [
        'top1_exact', 'top5_exact', 'top10_exact',
        'top1_scaf_match', 'top5_scaf_match', 'top10_scaf_match',
        'top1_mces', 'top5_mces', 'top10_mces',
        'top1_mces_dist', 'top5_mces_dist', 'top10_mces_dist',
        'top1_morgan', 'top5_morgan', 'top10_morgan',
        'top1_fg_sim', 'top5_fg_sim', 'top10_fg_sim',
        'top1_scaf_fp_sim', 'top5_scaf_fp_sim', 'top10_scaf_fp_sim'
    ]
    stats = {k: 0.0 for k in keys}

    for i in range(num_samples):
        # 1. Prepare GT Data
        gt_smi = label_lines[i]
        gt_mol = get_mol(gt_smi)
        gt_canon = get_canonical_smiles(gt_mol)
        gt_scaf_smi = get_murcko_scaffold_smiles(gt_mol)
        
        # Get GT functional groups
        gt_fgs_str = identify_func_groups(gt_mol)
        
        # 2. Prepare Top-1 Data
        top1_smi = top1_lines[i]
        top1_mol = get_mol(top1_smi)
        top1_canon = get_canonical_smiles(top1_mol)
        top1_scaf_smi = get_murcko_scaffold_smiles(top1_mol)
        
        # Get Pred Top-1 functional groups
        pred_top1_fgs_str = identify_func_groups(top1_mol)
        
        # 3. Prepare Top-10 Data List
        start_idx = i * num_preds_per_sample
        top10_chunk = top10_lines[start_idx : start_idx + num_preds_per_sample]
        
        top10_mols = [get_mol(s) for s in top10_chunk]
        # Precompute Top-10 properties for efficiency
        top10_props = []
        for m in top10_mols:
            if m:
                top10_props.append({
                    'mol': m,
                    'canon': get_canonical_smiles(m),
                    'scaf_smi': get_murcko_scaffold_smiles(m)
                })
        
        # 4. Prepare Top-5 Data List (Slice Top-10)
        top5_props = top10_props[:5]
        
        # --- Metric Calculation ---
        
        # 1. Exact Match (Accuracy)
        is_top1_match = 1 if (gt_canon and top1_canon and gt_canon == top1_canon) else 0
        is_top5_match = 1 if (gt_canon and any(p['canon'] == gt_canon for p in top5_props)) else 0
        is_top10_match = 1 if (gt_canon and any(p['canon'] == gt_canon for p in top10_props)) else 0
        
        # 2. Scaffold Match (Scaffold Accuracy)
        is_top1_scaf_match = 1 if (gt_scaf_smi and top1_scaf_smi and gt_scaf_smi == top1_scaf_smi) else 0
        is_top5_scaf_match = 1 if (gt_scaf_smi and any(p['scaf_smi'] == gt_scaf_smi for p in top5_props)) else 0
        is_top10_scaf_match = 1 if (gt_scaf_smi and any(p['scaf_smi'] == gt_scaf_smi for p in top10_props)) else 0
        
        # 3. MCES (Calculate Top-10 list, then take Max)
        # Top-1
        t1_mces_r, t1_mces_d = calculate_mces(gt_mol, top1_mol)
        
        # Calculate MCES for entire Top-10 list
        t10_mces_results = [calculate_mces(gt_mol, p['mol']) for p in top10_props]
        
        # Top-10 Best: Pick Max Ratio
        if t10_mces_results:
            t10_mces_r, t10_mces_d = max(t10_mces_results, key=lambda x: x[0])
        else:
            t10_mces_r = 0.0
            t10_mces_d = gt_mol.GetNumBonds() if gt_mol else 0
            
        # Top-5 Best (Slice first 5 then Max Ratio)
        t5_mces_results = t10_mces_results[:5]
        if t5_mces_results:
            t5_mces_r, t5_mces_d = max(t5_mces_results, key=lambda x: x[0])
        else:
            t5_mces_r = 0.0
            t5_mces_d = gt_mol.GetNumBonds() if gt_mol else 0

        # 4. Morgan Similarity
        t1_morgan = calculate_tanimoto_from_mols(gt_mol, top1_mol)
        t10_morgan_list = [calculate_tanimoto_from_mols(gt_mol, p['mol']) for p in top10_props]
        
        t10_morgan = max(t10_morgan_list) if t10_morgan_list else 0.0
        t5_morgan = max(t10_morgan_list[:5]) if t10_morgan_list[:5] else 0.0

        # 5. Functional Group Similarity
        t1_fg_sim = calculate_fg_similarity(gt_mol, top1_mol)
        t10_fg_list = [calculate_fg_similarity(gt_mol, p['mol']) for p in top10_props]
        
        t10_fg_sim = max(t10_fg_list) if t10_fg_list else 0.0
        t5_fg_sim = max(t10_fg_list[:5]) if t10_fg_list[:5] else 0.0

        # 6. Scaffold Fingerprint Similarity
        t1_scaf_fp_sim = calculate_scaffold_fingerprint_similarity(gt_mol, top1_mol)
        t10_scaf_fp_list = [calculate_scaffold_fingerprint_similarity(gt_mol, p['mol']) for p in top10_props]
        
        t10_scaf_fp_sim = max(t10_scaf_fp_list) if t10_scaf_fp_list else 0.0
        t5_scaf_fp_sim = max(t10_scaf_fp_list[:5]) if t10_scaf_fp_list[:5] else 0.0

        # --- Record Stats ---
        stats['top1_exact'] += is_top1_match
        stats['top5_exact'] += is_top5_match
        stats['top10_exact'] += is_top10_match
        
        stats['top1_scaf_match'] += is_top1_scaf_match
        stats['top5_scaf_match'] += is_top5_scaf_match
        stats['top10_scaf_match'] += is_top10_scaf_match
        
        stats['top1_mces'] += t1_mces_r
        stats['top5_mces'] += t5_mces_r
        stats['top10_mces'] += t10_mces_r
        
        stats['top1_mces_dist'] += t1_mces_d
        stats['top5_mces_dist'] += t5_mces_d
        stats['top10_mces_dist'] += t10_mces_d
        
        stats['top1_morgan'] += t1_morgan
        stats['top5_morgan'] += t5_morgan
        stats['top10_morgan'] += t10_morgan
        
        stats['top1_fg_sim'] += t1_fg_sim
        stats['top5_fg_sim'] += t5_fg_sim
        stats['top10_fg_sim'] += t10_fg_sim
        
        stats['top1_scaf_fp_sim'] += t1_scaf_fp_sim
        stats['top5_scaf_fp_sim'] += t5_scaf_fp_sim
        stats['top10_scaf_fp_sim'] += t10_scaf_fp_sim

        # Write result row
        results.append([
            gt_canon, top1_canon,
            gt_fgs_str, pred_top1_fgs_str, 
            is_top1_match, is_top5_match, is_top10_match,
            is_top1_scaf_match, is_top5_scaf_match, is_top10_scaf_match,
            f"{t1_mces_r:.4f}", f"{t5_mces_r:.4f}", f"{t10_mces_r:.4f}",
            f"{t1_mces_d:.1f}", f"{t5_mces_d:.1f}", f"{t10_mces_d:.1f}",
            f"{t1_morgan:.4f}", f"{t5_morgan:.4f}", f"{t10_morgan:.4f}",
            f"{t1_fg_sim:.4f}", f"{t5_fg_sim:.4f}", f"{t10_fg_sim:.4f}",
            f"{t1_scaf_fp_sim:.4f}", f"{t5_scaf_fp_sim:.4f}", f"{t10_scaf_fp_sim:.4f}"
        ])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples}...")

    # --- 5. Write CSV ---
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'GT_SMILES', 'Pred_Top1_SMILES',
                'GT_FuncGroups', 'Pred_Top1_FuncGroups', 
                'Top1_Exact_Acc', 'Top5_Exact_Acc', 'Top10_Exact_Acc',
                'Top1_Scaf_Acc', 'Top5_Scaf_Acc', 'Top10_Scaf_Acc',
                'Top1_Bond_Recall', 'Top5_Bond_Recall', 'Top10_Bond_Recall',
                'Top1_MCES_GED', 'Top5_MCES_GED', 'Top10_MCES_GED',
                'Top1_Morgan_Sim', 'Top5_Morgan_Sim', 'Top10_Morgan_Sim',
                'Top1_FG_Sim', 'Top5_FG_Sim', 'Top10_FG_Sim',
                'Top1_Scaf_Tanimoto_Sim', 'Top5_Scaf_Tanimoto_Sim', 'Top10_Scaf_Tanimoto_Sim'
            ])
            writer.writerows(results)
        print(f"CSV results saved to: {output_csv_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")
        return

    # --- 6. Generate and Save Report ---
    def avg(key): return stats[key] / num_samples if num_samples > 0 else 0

    report_lines = []
    report_lines.append("="*50)
    report_lines.append("              Comprehensive Metrics Report              ")
    report_lines.append("="*50)
    report_lines.append(f"Total Samples: {num_samples}")
    
    report_lines.append("\n--- Accuracy (Exact Match) ---")
    report_lines.append(f"Top-1  Accuracy: {avg('top1_exact')*100:.2f}%")
    report_lines.append(f"Top-5  Accuracy: {avg('top5_exact')*100:.2f}%")
    report_lines.append(f"Top-10 Accuracy: {avg('top10_exact')*100:.2f}%")
    
    report_lines.append("\n--- Scaffold Accuracy (Scaffold Exact Match) ---")
    report_lines.append(f"Top-1  Scaffold Accuracy: {avg('top1_scaf_match')*100:.2f}%")
    report_lines.append(f"Top-5  Scaffold Accuracy: {avg('top5_scaf_match')*100:.2f}%")
    report_lines.append(f"Top-10 Scaffold Accuracy: {avg('top10_scaf_match')*100:.2f}%")
    
    report_lines.append("\n--- MCES Metrics (Graph Edit Distance) ---")
    report_lines.append("Note: Lower MCES Distance is better; Higher Bond Recall is better")
    report_lines.append(f"Top-1  Avg Distance: {avg('top1_mces_dist'):.4f} (Bond Recall: {avg('top1_mces'):.4f})")
    report_lines.append(f"Top-5  Avg Distance: {avg('top5_mces_dist'):.4f} (Bond Recall: {avg('top5_mces'):.4f})")
    report_lines.append(f"Top-10 Avg Distance: {avg('top10_mces_dist'):.4f} (Bond Recall: {avg('top10_mces'):.4f})")
    
    report_lines.append("\n--- Similarity Metrics (Tanimoto / Jaccard) ---")
    report_lines.append(f"Top-1  Morgan Similarity: {avg('top1_morgan'):.4f}")
    report_lines.append(f"Top-5  Morgan Similarity: {avg('top5_morgan'):.4f}")
    report_lines.append(f"Top-10 Morgan Similarity: {avg('top10_morgan'):.4f}")
    
    report_lines.append(f"Top-1  Functional Group Similarity: {avg('top1_fg_sim'):.4f}")
    report_lines.append(f"Top-5  Functional Group Similarity: {avg('top5_fg_sim'):.4f}")
    report_lines.append(f"Top-10 Functional Group Similarity: {avg('top10_fg_sim'):.4f}")
    
    report_lines.append(f"Top-1  Scaffold Structure Similarity: {avg('top1_scaf_fp_sim'):.4f}")
    report_lines.append(f"Top-5  Scaffold Structure Similarity: {avg('top5_scaf_fp_sim'):.4f}")
    report_lines.append(f"Top-10 Scaffold Structure Similarity: {avg('top10_scaf_fp_sim'):.4f}")
    report_lines.append("="*50)

    # Print to console
    report_content = "\n".join(report_lines)
    print(report_content)

    # Save to TXT
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Metrics report saved to: {report_path}")
    except Exception as e:
        print(f"Failed to save report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate comprehensive metrics for predictions.")
    parser.add_argument("--label_file", required=True, help="Path to ground truth label file")
    parser.add_argument("--top10_file", required=True, help="Path to Top-10 prediction file")
    parser.add_argument("--top1_file", required=True, help="Path to Top-1 prediction file (filtered)")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    parser.add_argument("--report_file", required=True, help="Path to output report TXT")
    
    args = parser.parse_args()

    process_files(
        label_path=args.label_file,
        top10_path=args.top10_file,
        top1_path=args.top1_file,
        output_csv_path=args.output_csv,
        report_path=args.report_file
    )