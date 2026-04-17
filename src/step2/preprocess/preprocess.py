import os
import pickle
import copy
import glob
import random
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, BondType
from rdkit import RDLogger
from transforms import AddHigherOrderEdges, CountNodesPerGraph 

# Disable redundant RDKit logs
RDLogger.DisableLog('rdApp.*')

# --- Global Constants Definition ---
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# --- Helper Functions ---

def resample_spectrum(spectrum, target_len):
    spectrum_np = np.array(spectrum, dtype=np.float32)
    if spectrum_np.shape[0] == target_len:
        return torch.from_numpy(spectrum_np)
    else:
        resampled = resample(spectrum_np, target_len)
        return torch.tensor(resampled, dtype=torch.float32)

def canonicalize_smiles(smiles: str):
    smiles = smiles.replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def parse_spectrum_csv_with_smiles(file_path: str):
    """
    Reads CSV raw content only, does not construct molecules.
    Returns 3 values: smiles, pos, spectrum
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    smiles = lines[0]
    coord_lines, spectrum_parts = [], []
    is_coord_section = True
    
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) == 4 and is_coord_section:
            try:
                _ = [float(p) for p in parts]
                coord_lines.append(parts)
            except ValueError:
                is_coord_section = False
                spectrum_parts.extend(p for p in parts if p)
        else:
            is_coord_section = False
            spectrum_parts.extend(p for p in parts if p)
            
    if not coord_lines or not spectrum_parts:
        raise ValueError(f"File {file_path} format error: missing coords or spectrum.")

    coords = np.array([list(map(float, line)) for line in coord_lines])
    # Returns only pos (N, 3), not z
    pos = torch.tensor(coords[:, 1:], dtype=torch.float32)
    spectrum = [float(val) for val in spectrum_parts]
    
    return smiles, pos, spectrum

def create_mol_from_true_smiles(smiles: str, true_coords: np.ndarray) -> Mol:
    """[Train Set Mode] Create molecule based on true SMILES and true coordinates"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Verify number of atoms
        if mol.GetNumAtoms() != len(true_coords):
            return None
            
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(i, true_coords[i].tolist())
        
        mol.RemoveAllConformers()
        mol.AddConformer(conformer)
        return mol
    except Exception:
        return None

def create_mol_from_pred_smiles(smiles: str) -> Mol:
    """
    [Test Set Mode] Create molecule based on predicted SMILES and generate initial coarse conformation.
    This is fully end-to-end logic: no true coordinates, we use RDKit to generate initial coordinates.
    """
    try:
        if not smiles: return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        
        # Generate initial 3D conformation (ETKDG algorithm)
        # This step is to populate data.pos and ensure it matches molecular topology
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0:
            # If generation fails (very rare), fallback to computing 2D coords and setting Z=0
            AllChem.Compute2DCoords(mol)
        
        return mol
    except Exception:
        return None

def rdmol_to_data(mol: Mol) -> Data:
    assert mol.GetNumConformers() > 0
    N = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([start, end]); col.extend([end, start])
        edge_type.extend([BOND_TYPES.get(bond.GetBondType(), 0)] * 2)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    if edge_index.numel() > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index, edge_type = edge_index[:, perm], edge_type[perm]
    
    return Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, rdmol=copy.deepcopy(mol))

def load_test_specifications(id_file, smiles_file):
    print(f"📖 Reading test set definitions...")
    with open(id_file, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
        
    if len(ids) != len(smiles_list):
        raise ValueError(f"Error: ID file line count ({len(ids)}) matches SMILES file line count ({len(smiles_list)}) mismatch!")
    
    test_map = dict(zip(ids, smiles_list))
    print(f" > Loaded {len(test_map)} test sample definitions.")
    return test_map


import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess QMe14s dataset for SpecDiffMol.")
    parser.add_argument("--ir_dir", type=str, required=True, help="Directory containing IR spectra (e.g., dataset/qme14s/IR_broaden)")
    parser.add_argument("--raman_dir", type=str, required=True, help="Directory containing Raman spectra (e.g., dataset/qme14s/Raman_broaden)")
    parser.add_argument("--test_id_file", type=str, required=True, help="Path to test_indices.txt")
    parser.add_argument("--pred_smiles_file", type=str, required=True, help="Path to predicted SMILES file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--target_ir_len", type=int, default=3500, help="Target length for IR spectra")
    parser.add_argument("--target_raman_len", type=int, default=3500, help="Target length for Raman spectra")
    parser.add_argument("--max_molecules", type=int, default=0, help="Max molecules to process (0 for all)")
    parser.add_argument("--model_edge_order", type=int, default=3, help="Graph edge order")
    
    args = parser.parse_args()

    config = {
        'spectra_dirs': {
            'ir': args.ir_dir,
            'raman': args.raman_dir,
        },
        'test_id_file': args.test_id_file,          
        'pred_smiles_file': args.pred_smiles_file,
        'target_ir_len': args.target_ir_len,
        'target_raman_len': args.target_raman_len,
        'max_molecules': args.max_molecules if args.max_molecules > 0 else None, 
        'model_edge_order': args.model_edge_order,
        'output_dir': args.output_dir, 
    }

    print("--- Starting Dataset Preprocessing (End-to-End Real Mode) ---")

    # --- Step 0: Load Test Set Mapping ---
    test_prediction_map = load_test_specifications(config['test_id_file'], config['pred_smiles_file'])

    # --- Step 1: Define Graph Transformation Process ---
    graph_transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config['model_edge_order'])])

    # --- Step 2: Process Molecules ---
    train_data_list = []
    test_data_list = []
    
    ir_files = sorted(glob.glob(os.path.join(config['spectra_dirs']['ir'], 'IR_*.csv')))
    if config['max_molecules'] and len(ir_files) > config['max_molecules']:
        ir_files = ir_files[:config['max_molecules']]

    processed_count = 0
    
    for ir_path in tqdm(ir_files, desc=" > Processing"):
        file_id = os.path.basename(ir_path).replace('IR_', '').replace('.csv', '')
        raman_path = os.path.join(config['spectra_dirs']['raman'], f'Raman_{file_id}.csv')
        
        if not os.path.exists(raman_path):
            continue

        try:
            # 1. Read [True Data] as ground truth reference for both train and test
            # Note: parse function now returns only 3 values: smiles, pos, spectrum
            true_smiles_raw, true_pos, raw_ir_spec = parse_spectrum_csv_with_smiles(ir_path)
            
            # [Fix]: Changed to receive 3 variables
            _, _, raw_raman_spec = parse_spectrum_csv_with_smiles(raman_path)
            
            ir_spec = resample_spectrum(raw_ir_spec, config['target_ir_len'])
            raman_spec = resample_spectrum(raw_raman_spec, config['target_raman_len'])
            combined_spec = torch.cat([ir_spec, raman_spec], dim=0)
            
            clean_true_smiles = canonicalize_smiles(true_smiles_raw)

            # 2. Determine if it's train or test set, take different branches
            is_test_sample = (file_id in test_prediction_map)

            if is_test_sample:
                # ==========================================
                #   Test Set Logic: End-to-End Inference
                # ==========================================
                
                # A. Get predicted SMILES
                pred_smiles_raw = test_prediction_map[file_id]
                clean_pred_smiles = canonicalize_smiles(pred_smiles_raw)
                
                # Handle invalid SMILES
                target_smiles = clean_pred_smiles if clean_pred_smiles else pred_smiles_raw.replace(' ', '')
                
                # B. Build molecule graph and initial conformation based on PREDICTED SMILES
                mol = create_mol_from_pred_smiles(target_smiles)
                
                if mol is None:
                    print(f"Warning: ID {file_id} predicted SMILES could not build molecule, skipping.")
                    continue
                
                # C. Convert to graph data
                base_data = rdmol_to_data(mol)
                transformed_data = graph_transforms(base_data)
                
                # D. Fill attributes
                transformed_data.smiles = target_smiles  # This is the SMILES seen by the model
                transformed_data.ir_spectrum = ir_spec
                transformed_data.raman_spectrum = raman_spec
                transformed_data.combined_spectrum = combined_spec
                transformed_data.sample_id = torch.tensor([int(file_id)], dtype=torch.long)
                
                # E. Save true labels (renamed to avoid interference)
                transformed_data.true_smiles = clean_true_smiles
                transformed_data.true_pos_ref = true_pos 
                
                test_data_list.append(transformed_data)
                
            else:
                # ==========================================
                #   Train Set Logic: Oracle
                # ==========================================
                
                # A. Build molecule based on TRUE SMILES and TRUE POS
                mol = create_mol_from_true_smiles(clean_true_smiles, true_pos.numpy())
                if mol is None: continue

                # B. Convert to graph data
                base_data = rdmol_to_data(mol)
                transformed_data = graph_transforms(base_data)
                
                # C. Fill attributes
                transformed_data.smiles = clean_true_smiles
                transformed_data.ir_spectrum = ir_spec
                transformed_data.raman_spectrum = raman_spec
                transformed_data.combined_spectrum = combined_spec
                transformed_data.sample_id = torch.tensor([int(file_id)], dtype=torch.long)
                
                # Train set attributes
                transformed_data.true_smiles = clean_true_smiles
                
                train_data_list.append(transformed_data)
                
            processed_count += 1

        except Exception as e:
            # print(f"Error: Failed to process {file_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Processing complete.")
    print(f"   Total valid samples: {processed_count}")
    print(f"   Train set count: {len(train_data_list)}")
    print(f"   Test set count: {len(test_data_list)} (Constructed fictitious graph structure using predicted SMILES)")
    
    # --- Step 3: Save Dataset ---
    print(f"\n[Step 3/3] Saving dataset...")
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    sets_to_save = {
        'train.pkl': train_data_list,
        'test.pkl': test_data_list
    }

    for filename, dataset in sets_to_save.items():
        output_path = os.path.join(output_dir, filename)
        print(f"💾 Saving {filename} ({len(dataset)} samples) to '{output_path}'...")
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)

    print("\n🎉 All operations completed!")

if __name__ == '__main__':
    main()
    RDLogger.EnableLog('rdApp.*')