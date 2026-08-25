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

RDLogger.DisableLog("rdApp.*")

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}


def _make_atom(tok) -> Chem.Atom:
    tok = str(tok).strip()
    try:
        z = int(float(tok))
        if z > 0:
            return Chem.Atom(int(z))
    except (ValueError, TypeError):
        pass
    return Chem.Atom(tok)


def _to_symbol(tok) -> str:
    try:
        return _make_atom(tok).GetSymbol()
    except Exception:
        return str(tok).strip()


def normalize_and_resample(spectrum, target_len):
    s = np.array(spectrum, dtype=np.float32)
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo
    s = (s - lo) / rng if rng > 1e-8 else np.zeros_like(s)
    return torch.tensor(resample(s, target_len), dtype=torch.float32)


def canonicalize_smiles(smiles: str):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.replace(" ", ""))
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def parse_spectrum_csv_with_smiles(file_path: str):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    smiles = lines[0]
    coord_lines, spectrum_parts = [], []
    is_coord_section = True
    for line in lines[1:]:
        parts = line.split(",")
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

    elements = [_to_symbol(line[0]) for line in coord_lines]
    coords = np.array([list(map(float, line)) for line in coord_lines], dtype=np.float64)
    pos = torch.tensor(coords[:, 1:], dtype=torch.float32)
    spectrum = [float(v) for v in spectrum_parts]
    return smiles, elements, pos, spectrum


def create_mol_with_coords_from_smiles(smiles: str, elements, coords) -> Mol:
    try:
        coords = np.asarray(coords, dtype=np.float64)
        template = Chem.MolFromSmiles(smiles.replace(" ", "")) if smiles else None
        if template is None:
            return None
        template = Chem.AddHs(template)
        if template.GetNumAtoms() != len(elements) or template.GetNumAtoms() != len(coords):
            return None
        if sorted(a.GetSymbol() for a in template.GetAtoms()) != sorted(elements):
            return None
        query = Chem.RWMol()
        for elem in elements:
            query.AddAtom(_make_atom(elem))
        match = template.GetSubstructMatch(query.GetMol())
        if not match or len(match) != template.GetNumAtoms():
            return None
        conf = Chem.Conformer(template.GetNumAtoms())
        for query_idx, template_idx in enumerate(match):
            p = coords[query_idx]
            conf.SetAtomPosition(template_idx, [float(p[0]), float(p[1]), float(p[2])])
        final = copy.deepcopy(template)
        final.RemoveAllConformers()
        final.AddConformer(conf)
        return final
    except Exception:
        return None


def create_mol_from_pred_smiles(smiles: str) -> Mol:
    try:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def rdmol_to_data(mol: Mol) -> Data:
    assert mol.GetNumConformers() > 0
    N = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([s, e])
        col.extend([e, s])
        edge_type.extend([BOND_TYPES.get(bond.GetBondType(), 0)] * 2)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    if edge_index.numel() > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index, edge_type = edge_index[:, perm], edge_type[perm]
    return Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, rdmol=copy.deepcopy(mol))


def load_test_specifications(id_file, smiles_file):
    with open(id_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    if len(ids) != len(smiles_list):
        raise ValueError(f"ID count ({len(ids)}) does not match SMILES count ({len(smiles_list)})")
    print(f" > Loaded {len(ids)} test sample definitions.")
    return dict(zip(ids, smiles_list))


def main():
    config = {
        "spectra_dirs": {
            "ir": "../../dataset/qme14s/IR_broaden",
            "raman": "../../dataset/qme14s/Raman_broaden",
        },
        "test_id_file": "test_indices.txt",
        "pred_smiles_file": "result.txt",
        "target_ir_len": 3500,
        "target_raman_len": 3500,
        "max_molecules": 10000000,
        "model_edge_order": 3,
        "output_dir": "./data",
        "val_ratio": 0.1,
        "seed": 42,
    }

    print("--- QMe14s Preprocessing ---")
    test_map = load_test_specifications(config["test_id_file"], config["pred_smiles_file"])
    graph_transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config["model_edge_order"])])

    train_list, test_list = [], []
    ir_files = sorted(glob.glob(os.path.join(config["spectra_dirs"]["ir"], "IR_*.csv")))
    if config["max_molecules"] and len(ir_files) > config["max_molecules"]:
        ir_files = ir_files[: config["max_molecules"]]

    processed, skip_recon, skip_pred = 0, 0, 0
    for ir_path in tqdm(ir_files, desc=" > Processing"):
        file_id = os.path.basename(ir_path).replace("IR_", "").replace(".csv", "")
        raman_path = os.path.join(config["spectra_dirs"]["raman"], f"Raman_{file_id}.csv")
        if not os.path.exists(raman_path):
            continue
        try:
            true_smiles_raw, elements, true_pos, raw_ir = parse_spectrum_csv_with_smiles(ir_path)
            _, _, _, raw_raman = parse_spectrum_csv_with_smiles(raman_path)

            ir_spec = normalize_and_resample(raw_ir, config["target_ir_len"])
            raman_spec = normalize_and_resample(raw_raman, config["target_raman_len"])
            combined_spec = torch.cat([ir_spec, raman_spec], dim=0)

            true_smiles_canon = canonicalize_smiles(true_smiles_raw)

            if file_id in test_map:
                # Test set: model graph uses the predicted SMILES; reference conformer is rebuilt from the true SMILES
                pred_raw = test_map[file_id]
                pred_canon = canonicalize_smiles(pred_raw)
                target_smiles = pred_canon if pred_canon else pred_raw.replace(" ", "")

                mol_graph = create_mol_from_pred_smiles(target_smiles)
                if mol_graph is None:
                    skip_pred += 1
                    continue
                true_ref = create_mol_with_coords_from_smiles(true_smiles_raw, elements, true_pos.numpy())
                if true_ref is None:
                    skip_recon += 1
                    continue

                data = graph_transforms(rdmol_to_data(mol_graph))
                data.smiles = target_smiles
                data.ir_spectrum = ir_spec
                data.raman_spectrum = raman_spec
                data.combined_spectrum = combined_spec
                data.sample_id = torch.tensor([int(file_id)], dtype=torch.long)
                data.true_smiles = true_smiles_canon
                data.true_rdmol = true_ref
                test_list.append(data)
            else:
                # Train set: true SMILES + true coordinates
                mol = create_mol_with_coords_from_smiles(true_smiles_raw, elements, true_pos.numpy())
                if mol is None:
                    skip_recon += 1
                    continue
                data = graph_transforms(rdmol_to_data(mol))
                data.smiles = true_smiles_canon
                data.ir_spectrum = ir_spec
                data.raman_spectrum = raman_spec
                data.combined_spectrum = combined_spec
                data.sample_id = torch.tensor([int(file_id)], dtype=torch.long)
                data.true_smiles = true_smiles_canon
                train_list.append(data)

            processed += 1
        except Exception:
            import traceback

            traceback.print_exc()
            continue

    print(f"Done. Valid: {processed} | Train (pre-split): {len(train_list)} | Test: {len(test_list)}")
    print(
        f"   Skipped (reconstruction failed): {skip_recon} | Skipped (predicted SMILES graph build failed): {skip_pred}"
    )

    rng = random.Random(config["seed"])
    rng.shuffle(train_list)
    n_val = max(1, int(round(len(train_list) * config["val_ratio"]))) if train_list else 0
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]
    print(
        f"Train/Val split ({1 - config['val_ratio']:.0%}/{config['val_ratio']:.0%}): "
        f"Train {len(train_list)} | Val {len(val_list)}"
    )

    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)
    for fname, ds in {"train.pkl": train_list, "val.pkl": val_list, "test.pkl": test_list}.items():
        p = os.path.join(out, fname)
        print(f"Saving {fname} ({len(ds)} samples) -> {p}")
        with open(p, "wb") as f:
            pickle.dump(ds, f)
    print("All done.")


if __name__ == "__main__":
    main()
    RDLogger.EnableLog("rdApp.*")
