import os
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from collections import Counter

import click
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from scipy.signal import resample, find_peaks 
from sklearn.model_selection import train_test_split
import regex as re

try:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula 
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Error: RDKit is required for structural parsing. Please install it.")
    exit(1)

try:
    from rxn.chemutils.tokenization import tokenize_smiles
except ImportError:
    # Use a simple fallback if rxn-chemutils is not available
    def tokenize_smiles(smiles): return ' '.join(list(smiles)) 


def normalize_and_resample(spectrum: List[float], target_len: int) -> np.ndarray:
    """Normalize spectrum (0-1) and resample to target length."""
    spectrum_np = np.array(spectrum, dtype=np.float32)
    min_val = spectrum_np.min()
    max_val = spectrum_np.max()
    range_val = max_val - min_val
    if range_val > 1e-8:
        spectrum_np = (spectrum_np - min_val) / range_val
    else:
        spectrum_np = np.zeros_like(spectrum_np)
        
    resampled = resample(spectrum_np, target_len)
    return resampled.astype(np.float32)


def extract_peaks_and_format(
    spectrum_np: np.ndarray, 
    prefix: str, 
    peak_min_height: float = 0.05, 
    max_peaks: int = 100
) -> str:
    """
    Extract peaks from preprocessed array and format as [Index Intensity] sequence.
    Both index and intensity are formatted to 1 decimal place.
    """
    if spectrum_np.size == 0:
        return f"{prefix} 0.0 0.0"

    # Scale 0-1 to 0-100
    spectrum_100 = spectrum_np * 100
    
    max_intensity = np.max(spectrum_100)
    min_height_absolute = max_intensity * peak_min_height
    
    peaks_indices, properties = find_peaks(
        spectrum_100, 
        height=min_height_absolute,
    )
    
    if len(peaks_indices) == 0:
        return f"{prefix} 0.0 0.0"

    peaks_intensities = properties['peak_heights']

    # Sort by intensity descending
    peak_data = sorted(
        zip(peaks_indices, peaks_intensities),
        key=lambda x: x[1],
        reverse=True
    )
    
    peak_data = peak_data[:max_peaks]

    tokens = []
    for index, intensity in peak_data:
        tokens.append(f"{index:.1f}")
        tokens.append(f"{intensity:.1f}") 

    return f'{prefix} ' + ' '.join(tokens)


def parse_spectrum_csv_with_smiles(file_path: str):
    """Parse qme14s CSV file, returning SMILES, atom numbers, and spectrum data."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    smiles = lines[0]
    
    coord_lines = []
    spectrum_parts = []
    
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
    z = coords[:, 0].astype(int) 
    spectrum = [float(val) for val in spectrum_parts]
    
    return smiles, z, spectrum


def tokenize_formula(formula: str) -> str:
    """Convert formula string (e.g. "C4H6O") to tokenized form (e.g. "C 4 H 6 O")."""
    return ' '.join(re.findall("[A-Z][a-z]?|\d+", formula)).strip()


def process_qme14s_files(
    ir_dir: Path, 
    raman_dir: Path, 
    base_spectra_len: int,
    peak_min_height: float,
    max_peaks: int,
    max_samples: int = 0
) -> pd.DataFrame:
    """Read and process data from CSV files."""
    
    input_list = list()
    
    ir_files = sorted(glob.glob(str(ir_dir / 'IR_*.csv')))
    
    if max_samples > 0 and len(ir_files) > max_samples:
        ir_files = ir_files[:max_samples]

    print(f"Found {len(ir_files)} potential IR files to process.")

    for ir_path_str in tqdm(ir_files, desc="Processing qme14s data"):
        ir_path = Path(ir_path_str)
        file_id = ir_path.name.replace('IR_', '').replace('.csv', '')
        raman_path = raman_dir / f'Raman_{file_id}.csv'
        
        if not raman_path.exists():
            continue 

        try:
            # 1. Parse files
            smiles, z_ir, raw_ir_spec = parse_spectrum_csv_with_smiles(str(ir_path))
            _, z_raman, raw_raman_spec = parse_spectrum_csv_with_smiles(str(raman_path))
            
            if not np.array_equal(z_ir, z_raman):
                 continue
            
            # 2. Structural processing and validity check
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            tokenized_target = tokenize_smiles(canonical_smiles) 

            # 3. Generate Source parts
            
            # 3a. Formula (Optional, currently commented out in original logic)
            # mol_h = Chem.AddHs(mol) 
            # formula_str = CalcMolFormula(mol_h)
            tokenized_formula =''
            
            # 3b. IR Peaks
            ir_resampled = normalize_and_resample(raw_ir_spec, base_spectra_len)
            tokenized_ir = extract_peaks_and_format(
                ir_resampled, 'IR_PEAKS', peak_min_height, max_peaks
            )
            
            # 3c. Raman Peaks
            raman_resampled = normalize_and_resample(raw_raman_spec, base_spectra_len)
            tokenized_raman = extract_peaks_and_format(
                raman_resampled, 'RAMAN_PEAKS', peak_min_height, max_peaks
            )
            
            # 4. Combine Source string
            tokenized_input = f"{tokenized_formula} {tokenized_ir} {tokenized_raman}"
            
            input_list.append({
                'source': tokenized_input.strip(), 
                'target': tokenized_target.strip(),
                'canonical_smiles': canonical_smiles,
                'file_id': file_id
            })
        
        except Exception:
            continue
    
    input_df = pd.DataFrame(input_list)
    
    # Deduplicate based on canonical SMILES
    if not input_df.empty:
        input_df = input_df.drop_duplicates(subset="canonical_smiles", keep='first')
        input_df = input_df.drop(columns=['canonical_smiles'])
        
    return input_df


def split_data(data: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    train, test = train_test_split(data, test_size=0.1, random_state=seed, shuffle=True)
    return train, test


def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str) -> None:
    """Save processed DataFrame as src/tgt text files for Transformer."""
    out_path.mkdir(parents=True, exist_ok=True)

    sources = list(data_set.source)
    targets = list(data_set.target)

    # Source = FORMULA + Spectrum Peaks
    with (out_path / f"src-{set_type}.txt").open("w") as f:
        for item in sources:
            f.write(f"{item}\n")
    
    # Target = Tokenized SMILES
    with (out_path / f"tgt-{set_type}.txt").open("w") as f:
        for item in targets:
            f.write(f"{item}\n")


def save_indices(data_set: pd.DataFrame, out_path: Path, set_type: str) -> None:
    """Save file_id indices for training/test sets."""
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_ids = list(data_set.file_id)
    
    file_name = f"{set_type}_indices.txt"
    with (out_path / file_name).open("w") as f:
        for index in file_ids:
            f.write(f"{index}\n")
    print(f"   > Saved indices to {out_path / file_name}")


@click.command()
@click.option(
    "--ir_data_dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the directory containing IR_*.csv files.",
)
@click.option(
    "--raman_data_dir",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the directory containing Raman_*.csv files.",
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    default='./processed_data',
    help="Output path to save the 'data' directory.",
)
@click.option(
    "--base_spectra_len",
    type=int,
    default=600,
    show_default=True,
    help="Target length for resampling.",
)
@click.option(
    "--peak_min_height",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum peak height (0.0-1.0).",
)
@click.option(
    "--max_peaks",
    type=int,
    default=100,
    show_default=True,
    help="Max number of peaks to extract.",
)
@click.option(
    "--max_samples",
    type=int,
    default=0,
    show_default=True,
    help="Max number of molecules to process. 0 means all.",
)
@click.option(
    "--seed", 
    type=int, 
    default=3245, 
    show_default=True,
    help="Random seed."
)
def main(
    ir_data_dir: Path,
    raman_data_dir: Path,
    out_path: Path,
    base_spectra_len: int,
    peak_min_height: float,
    max_peaks: int,
    max_samples: int,
    seed: int
): 
    """
    Main Preprocessing Script: extracts formula and spectrum peaks from QMe14s data,
    saves as transformer training files.
    """
    
    print("--- Starting QMe14s Preprocessing ---")
    
    tokenised_data_df = process_qme14s_files(
        ir_data_dir, 
        raman_data_dir, 
        base_spectra_len,
        peak_min_height,
        max_peaks,
        max_samples
    )
    
    if tokenised_data_df.empty:
        print("Error: No data was successfully processed.")
        return
        
    print(f"Successfully processed {len(tokenised_data_df)} unique entries.")

    train_set, test_set = split_data(tokenised_data_df, seed)
    print(f"Split into {len(train_set)} training and {len(test_set)} test samples.")

    out_data_path = out_path / "data"
    print(f"Saving files to {out_data_path}...")
    
    save_set(test_set, out_data_path, "test")
    save_set(train_set, out_data_path, "train")
    
    save_indices(test_set, out_data_path, "test")
    save_indices(train_set, out_data_path, "train")
    
    print("\nPre-processing complete!")


if __name__ == '__main__':
    main()