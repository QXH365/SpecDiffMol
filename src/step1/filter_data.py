from rdkit import Chem
from rdkit import RDLogger
from collections import Counter
import argparse

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def get_molecular_formula(smiles_string):
    """
    Get molecular formula from SMILES string (as element count dictionary, heavy atoms only, no hydrogen)
    """
    try:
        cleaned_smiles = "".join(smiles_string.strip().split())
        mol = Chem.MolFromSmiles(cleaned_smiles)
        if mol is None:
            return None
        
        # Get formula as dictionary (heavy atoms only, exclude H)
        formula = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol != 'H':  # Exclude Hydrogen
                formula[symbol] = formula.get(symbol, 0) + 1
        return formula
    except:
        return None

def is_valid_smiles(smiles_string):
    """
    Check if SMILES is valid
    """
    try:
        cleaned_smiles = "".join(smiles_string.strip().split())
        mol = Chem.MolFromSmiles(cleaned_smiles)
        return mol is not None
    except:
        return False

def formulas_match(formula1, formula2):
    """
    Check if two molecular formulas match
    """
    if formula1 is None or formula2 is None:
        return False
    return formula1 == formula2

def get_heavy_atom_count(formula):
    """
    Calculate total heavy atom count from formula dictionary
    """
    if formula is None:
        return -1
    return sum(formula.values())

def process_predictions(input_path, output_path, label_path, n_best=10, output_n_best=5):
    """
    Process prediction results file
    """
    print(f"Starting prediction file processing...")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Label file: {label_path}")
    
    # Read all lines
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            pred_lines = f.readlines()
        
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return
    
    # Verify line counts
    num_samples = len(label_lines)
    expected_pred_lines = num_samples * n_best
    
    if len(pred_lines) != expected_pred_lines:
        print(f"Warning: Prediction file line count mismatch!")
        print(f"Label file: {num_samples} samples")
        print(f"Prediction file: {len(pred_lines)} lines (Expected {expected_pred_lines} lines)")
        print(f"Processing based on actual line count...")
        num_samples = min(num_samples, len(pred_lines) // n_best)
    
    print(f"Processing {num_samples} samples...")
    
    output_lines = []
    stats = {
        'total_samples': 0,
        'valid_formula_match': 0,     # Priority 1
        'valid_atom_count_match': 0,  # Priority 2
        'valid_others': 0,            # Priority 3
        'invalid_used': 0,            # Priority 4
        'insufficient_candidates': 0
    }
    
    # Process sample by sample
    for i in range(num_samples):
        # Get label formula and heavy atom count
        label_line = label_lines[i].strip()
        target_formula = get_molecular_formula(label_line)
        target_heavy_count = get_heavy_atom_count(target_formula)
        
        # Get n_best predictions for this sample
        start_idx = i * n_best
        end_idx = start_idx + n_best
        predictions = pred_lines[start_idx:end_idx]
        
        # Define four priority lists
        p1_formula_match = []      # Priority 1: Valid and formula matches exactly
        p2_count_match = []        # Priority 2: Valid and heavy atom count matches (but formula differs)
        p3_valid_others = []       # Priority 3: Valid but heavy atom count differs
        p4_invalid = []            # Priority 4: Invalid SMILES
        
        for pred_line in predictions:
            pred_smiles = pred_line.strip()
            
            # Check validity first
            if is_valid_smiles(pred_smiles):
                pred_formula = get_molecular_formula(pred_smiles)
                pred_heavy_count = get_heavy_atom_count(pred_formula)
                
                # Priority judgment
                if formulas_match(target_formula, pred_formula):
                    p1_formula_match.append(pred_smiles)
                elif pred_heavy_count == target_heavy_count and target_heavy_count > 0:
                    # Formula mismatch but heavy atom count matches
                    p2_count_match.append(pred_smiles)
                else:
                    # Valid but neither formula nor atom count matches
                    p3_valid_others.append(pred_smiles)
            else:
                p4_invalid.append(pred_smiles)
        
        # Combine results by priority
        filtered_results = []
        
        # 1. Prioritize formula match
        filtered_results.extend(p1_formula_match)
        
        # 2. If insufficient, use heavy atom count match
        if len(filtered_results) < output_n_best:
            remaining = output_n_best - len(filtered_results)
            filtered_results.extend(p2_count_match[:remaining])
            
        # 3. If insufficient, use other valid ones
        if len(filtered_results) < output_n_best:
            remaining = output_n_best - len(filtered_results)
            filtered_results.extend(p3_valid_others[:remaining])
        
        # 4. If still insufficient, use invalid ones
        if len(filtered_results) < output_n_best:
            remaining = output_n_best - len(filtered_results)
            filtered_results.extend(p4_invalid[:remaining])
        
        # 5. If still insufficient (extreme case), repeat the last one
        while len(filtered_results) < output_n_best:
            if filtered_results:
                filtered_results.append(filtered_results[-1])
            else:
                # If none at all, use original first
                filtered_results.append(predictions[0].strip() if predictions else "")
        
        # Truncate to top output_n_best
        filtered_results = filtered_results[:output_n_best]
        
        # Add to output
        output_lines.extend([result + '\n' for result in filtered_results])
        
        # --- Stats Logic ---
        stats['total_samples'] += 1
        
        # Count number of each category in selected results (Simplified stats)
        stats['valid_formula_match'] += len(p1_formula_match)
        stats['valid_atom_count_match'] += len(p2_count_match)
        stats['valid_others'] += len(p3_valid_others)
        
        # Check if forced to use invalid SMILES
        current_valid_count = len(p1_formula_match) + len(p2_count_match) + len(p3_valid_others)
        if current_valid_count < output_n_best:
             stats['invalid_used'] += 1
             stats['insufficient_candidates'] += 1
        
        # Print progress
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")
    
    # Write output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
    except IOError as e:
        print(f"Error: Could not write file {output_path}")
        return
    
    # Print stats
    print("\n=== Processing Complete ===")
    print(f"Output file: {output_path}")
    print(f"Total lines: {len(output_lines)}")
    
    print("\n=== Candidate Quality Stats (Avg per sample) ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"1. Formula Exact Match: {stats['valid_formula_match'] / stats['total_samples']:.2f}")
    print(f"2. Heavy Atom Count Match Only: {stats['valid_atom_count_match'] / stats['total_samples']:.2f}")
    print(f"3. Other Valid Molecules: {stats['valid_others'] / stats['total_samples']:.2f}")
    
    print("\n=== Anomaly Stats ===")
    print(f"Samples forced to use invalid SMILES: {stats['invalid_used']}")
    
    # Calculate filter quality
    total_valid_candidates = stats['valid_formula_match'] + stats['valid_atom_count_match'] + stats['valid_others']
    if total_valid_candidates > 0:
        match_rate = stats['valid_formula_match'] / total_valid_candidates * 100
        count_match_rate = stats['valid_atom_count_match'] / total_valid_candidates * 100
        print(f"\nValid Candidate Distribution:")
        print(f"- Formula Match: {match_rate:.2f}%")
        print(f"- Atom Count Match: {count_match_rate:.2f}%")
        print(f"- Other Valid: {100 - match_rate - count_match_rate:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter prediction results.")
    parser.add_argument("--input_file", required=True, help="Input prediction file")
    parser.add_argument("--output_file", required=True, help="Output filtered file")
    parser.add_argument("--label_file", required=True, help="Label file for target formula")
    parser.add_argument("--n_best", type=int, default=10, help="Candidates per sample in input")
    parser.add_argument("--output_n_best", type=int, default=1, help="Candidates per sample in output")
    
    args = parser.parse_args()

    process_predictions(
        input_path=args.input_file,
        output_path=args.output_file,
        label_path=args.label_file,
        n_best=args.n_best,
        output_n_best=args.output_n_best
    )