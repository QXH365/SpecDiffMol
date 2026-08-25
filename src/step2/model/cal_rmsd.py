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

RDLogger.DisableLog("rdApp.*")


def _prep_mol(mol, heavy_only: bool):
    """Preprocess a molecule (strip hydrogens).
    Tolerates molecules with slightly inconsistent valence perception (e.g. a
    geometry-inferred bond that pushes carbon to valence 5), so that RemoveHs'
    sanitize valence check doesn't raise AtomValenceException and abort the
    whole batch."""
    if mol is None:
        return None
    m = Chem.Mol(mol)
    if not heavy_only:
        return m
    try:
        return Chem.RemoveHs(m)
    except Exception:
        try:
            m2 = Chem.RemoveHs(m, sanitize=False)
            m2.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(m2)
            return m2
        except Exception:
            return None


def _coords(mol):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    return np.asarray(pos, dtype=float)


def _rmsd_in_frame(all_pairs, gen_xyz, ref_xyz):
    """Compute RMSD over the pair set directly in the current frame (gen_xyz is
    already aligned)."""
    sq = 0.0
    for gi, rj in all_pairs:
        diff = gen_xyz[gi] - ref_xyz[rj]
        sq += float(diff.dot(diff))
    return float(np.sqrt(sq / len(all_pairs)))


def get_mol_props(smiles):
    if not smiles:
        return None, -1

    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, -1

    formula = rdMolDescriptors.CalcMolFormula(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    return formula, heavy_atoms


def standardize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    s = smiles.replace(" ", "")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ""
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def mol_to_coords_str(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return ""

    conf = mol.GetConformer()
    coords = conf.GetPositions()

    return "; ".join([f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}" for row in coords])


def mcs_align_plus_hungarian(
    ref_mol,
    gen_mol,
    heavy_only=True,
    mcs_timeout=10,
    max_matches=30,
    max_pair_trials=400,
    element_mismatch_penalty=1e6,
    allow_atom_mismatch=False,
    final_align_full=False,
):
    if ref_mol is None or gen_mol is None:
        return None, None, 0, 0, "no_mol", 0.0

    ref = _prep_mol(ref_mol, heavy_only)
    gen = _prep_mol(gen_mol, heavy_only)

    if ref is None or gen is None:
        return None, None, 0, 0, "prep_mol_failed", 0.0

    if ref.GetNumConformers() == 0 or gen.GetNumConformers() == 0:
        return None, None, 0, 0, "no_conformer", 0.0

    n_ref = ref.GetNumAtoms()
    n_gen = gen.GetNumAtoms()
    is_partial = n_ref != n_gen
    if is_partial and not allow_atom_mismatch:
        return None, None, 0, 0, "atom_count_mismatch", 0.0

    denom = max(n_ref, n_gen)

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
        return None, None, 0, 0, "mcs_exception", 0.0

    if not mcs or mcs.numAtoms <= 0 or not mcs.smartsString:
        return None, None, 0, 0, "mcs_not_found", 0.0

    q = Chem.MolFromSmarts(mcs.smartsString)
    if q is None:
        return None, None, 0, 0, "mcs_smarts_bad", 0.0

    ref_matches = ref.GetSubstructMatches(q, uniquify=False, maxMatches=max_matches)
    gen_matches = gen.GetSubstructMatches(q, uniquify=False, maxMatches=max_matches)
    if not ref_matches or not gen_matches:
        return None, None, int(mcs.numAtoms), 0, "mcs_match_empty", 0.0

    ref_symbols = [a.GetSymbol() for a in ref.GetAtoms()]
    gen_symbols = [a.GetSymbol() for a in gen.GetAtoms()]
    ok_status = "ok_partial" if is_partial else "ok"
    best = {
        "rmsd_total": None,
        "rmsd_mcs": None,
        "mcs_atoms": int(mcs.numAtoms),
        "n_extra": 0,
        "status": "failed",
        "coverage": 0.0,
    }

    trials = 0
    for rm in ref_matches:
        rm_set = set(rm)
        for gm in gen_matches:
            trials += 1
            if trials > max_pair_trials:
                break

            gen_copy = Chem.Mol(gen)
            atom_map = list(zip(gm, rm))  # (gen_idx, ref_idx)

            try:
                rmsd_mcs = float(rdMolAlign.AlignMol(gen_copy, ref, atomMap=atom_map))
            except Exception:
                continue

            gm_set = set(gm)

            u_ref = [i for i in range(n_ref) if i not in rm_set]
            u_gen = [i for i in range(n_gen) if i not in gm_set]

            if not is_partial and len(u_ref) != len(u_gen):
                continue

            try:
                ref_xyz = _coords(ref)
                gen_xyz = _coords(gen_copy)
            except Exception:
                continue

            n_ur, n_ug = len(u_ref), len(u_gen)
            extra_pairs = []
            if n_ur > 0 and n_ug > 0:
                cost = np.zeros((n_ug, n_ur), dtype=float)
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
                if is_partial:
                    for ii, jj in zip(row_ind, col_ind):
                        gi, rj = u_gen[ii], u_ref[jj]
                        if gen_symbols[gi] != ref_symbols[rj]:
                            continue
                        extra_pairs.append((gi, rj))
                else:
                    ok = True
                    for ii, jj in zip(row_ind, col_ind):
                        gi, rj = u_gen[ii], u_ref[jj]
                        if gen_symbols[gi] != ref_symbols[rj]:
                            ok = False
                            break
                        extra_pairs.append((gi, rj))
                    if not ok:
                        continue

            # Pair set P: MCS pairs + Hungarian-completed pairs (a partial
            # match does not cover the whole molecule).
            all_pairs = atom_map + extra_pairs
            if not all_pairs:
                continue
            if final_align_full:
                try:
                    gen_final = Chem.Mol(gen)
                    rmsd_total = float(rdMolAlign.AlignMol(gen_final, ref, atomMap=all_pairs))
                except Exception:
                    rmsd_total = _rmsd_in_frame(all_pairs, gen_xyz, ref_xyz)
            else:
                rmsd_total = _rmsd_in_frame(all_pairs, gen_xyz, ref_xyz)
            if best["rmsd_total"] is None or rmsd_total < best["rmsd_total"]:
                best.update(
                    {
                        "rmsd_total": rmsd_total,
                        "rmsd_mcs": rmsd_mcs,
                        "mcs_atoms": mcs.numAtoms,
                        "n_extra": len(extra_pairs),
                        "status": ok_status,
                        "coverage": len(all_pairs) / denom,
                    }
                )

        if trials > max_pair_trials:
            break

    if best["status"] not in ("ok", "ok_partial"):
        return None, None, int(mcs.numAtoms), 0, "align_or_hungarian_failed", 0.0

    return (best["rmsd_total"], best["rmsd_mcs"], best["mcs_atoms"], best["n_extra"], best["status"], best["coverage"])


def main():
    parser = argparse.ArgumentParser(description="Evaluation script: MCS + Hungarian method")
    parser.add_argument("--result_pkl", type=str, default="./inference_results_old.pkl", help="Inference pickle file")
    parser.add_argument(
        "--output_csv", type=str, default="analysis_report_mcs_hungarian.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--use_all_atoms",
        action="store_true",
        help="Compute RMSD over all atoms including H (default: heavy atoms only)",
    )
    parser.add_argument("--mcs_timeout", type=int, default=10, help="MCS search timeout (seconds)")
    parser.add_argument("--max_matches", type=int, default=30, help="MCS max number of matches")
    parser.add_argument("--max_pair_trials", type=int, default=400, help="MCS max number of match-pair trials")
    parser.add_argument(
        "--allow_atom_mismatch",
        action="store_true",
        help="Also force a partial RMSD for samples whose heavy-atom counts differ "
        "(status=ok_partial); always read it together with Match Coverage, and "
        "never mix it with the strict whole-molecule RMSD",
    )
    parser.add_argument(
        "--final_align_full",
        action="store_true",
        help="Report the final RMSD after an extra Kabsch alignment over the full "
        "pair set P (default: computed directly in the MCS core frame). "
        "Enable this to remove RMSD inflation caused by low coverage.",
    )
    args = parser.parse_args()

    print(f"Loading inference results: {args.result_pkl} ...")
    if not os.path.exists(args.result_pkl):
        print("Error: file not found")
        return

    with open(args.result_pkl, "rb") as f:
        data_list = pickle.load(f)

    heavy_only = not args.use_all_atoms

    print(f"Loaded {len(data_list)} samples")
    print(f"RMSD calculation mode: {'All atoms (with H)' if args.use_all_atoms else 'Heavy atoms only'}")
    print("Evaluation strategy: MCS + Hungarian")

    csv_rows = []

    stats = {
        "total_samples": len(data_list),
        "total_confs": 0,
        "valid_rmsd_count": 0,
        "partial_rmsd_count": 0,
        "atom_count_mismatch": 0,
        "failed_count": 0,
    }

    all_valid_rmsds = []
    all_mcs_coverage = []
    all_extra_atoms = []
    all_partial_rmsds = []
    all_match_coverage = []

    for i, item in tqdm(enumerate(data_list), total=len(data_list), desc="Processing"):
        true_smiles = item.get("true_smiles", "") or ""
        pred_smiles = item.get("pred_smiles", "") or ""
        true_mol = item.get("true_mol", None)
        gen_mols = item.get("gen_mols", None) or []

        # 1. Basic property check
        smi_correct = (
            (true_smiles.replace(" ", "") == pred_smiles.replace(" ", "")) if (true_smiles and pred_smiles) else False
        )
        true_formula, true_heavy = get_mol_props(true_smiles)
        pred_formula, pred_heavy = get_mol_props(pred_smiles)
        formula_match = (true_formula == pred_formula) if true_formula else False
        heavy_atom_match = (true_heavy == pred_heavy) if (true_heavy != -1) else False
        true_std = standardize_smiles(true_smiles)
        pred_std = standardize_smiles(pred_smiles)
        pos_true_str = mol_to_coords_str(true_mol)

        # 2. Process the generated conformers
        if not gen_mols:
            csv_rows.append(
                {
                    "Index": f"{i}_Err",
                    "True SMILES": true_std,
                    "Predicted SMILES": pred_std,
                    "SMILES Correct": smi_correct,
                    "Formula Match": formula_match,
                    "Heavy Atom Match": heavy_atom_match,
                    "RMSD (Å)": "",
                    "Note": "No generated mols",
                    "Status": "no_gen_mols",
                }
            )
            continue

        # Denominator: true atom count (heavy or all atoms)
        true_for_cov = _prep_mol(true_mol, heavy_only) if true_mol is not None else None
        n_true_atoms = true_for_cov.GetNumAtoms() if true_for_cov is not None else 0

        best_rmsd_total = None
        best_rmsd_mcs = None
        best_conf_id = None
        best_mcs_atoms = 0
        best_extra = 0
        best_coverage = 0.0
        status = "no_true_mol" if true_mol is None else "failed"

        if true_mol is not None:
            for k, gm in enumerate(gen_mols):
                stats["total_confs"] += 1

                try:
                    rmsd_total, rmsd_mcs, mcs_atoms, n_extra, st, cov = mcs_align_plus_hungarian(
                        true_mol,
                        gm,
                        heavy_only=heavy_only,
                        mcs_timeout=args.mcs_timeout,
                        max_matches=args.max_matches,
                        max_pair_trials=args.max_pair_trials,
                        allow_atom_mismatch=args.allow_atom_mismatch,
                        final_align_full=args.final_align_full,
                    )
                except Exception:
                    rmsd_total, rmsd_mcs, mcs_atoms, n_extra, st, cov = None, None, 0, 0, "exception", 0.0

                if st == "atom_count_mismatch":
                    stats["atom_count_mismatch"] += 1
                    status = "atom_count_mismatch"
                    continue

                if st not in ("ok", "ok_partial") or rmsd_total is None:
                    status = st
                    continue

                if best_rmsd_total is None or rmsd_total < best_rmsd_total:
                    best_rmsd_total = rmsd_total
                    best_rmsd_mcs = rmsd_mcs
                    best_conf_id = k
                    best_mcs_atoms = int(mcs_atoms)
                    best_extra = int(n_extra)
                    best_coverage = float(cov)
                    status = st  # "ok" or "ok_partial"

        is_success = status in ("ok", "ok_partial")

        mcs_pct = ""
        match_pct = ""
        if is_success and n_true_atoms > 0:
            mcs_pct = f"{(best_mcs_atoms / n_true_atoms * 100.0):.2f}"
            all_mcs_coverage.append(best_mcs_atoms / n_true_atoms * 100.0)
            all_extra_atoms.append(best_extra)
        if is_success:
            match_pct = f"{best_coverage * 100.0:.2f}"

        if status == "ok":
            stats["valid_rmsd_count"] += 1
            all_valid_rmsds.append(best_rmsd_total)
        elif status == "ok_partial":
            stats["partial_rmsd_count"] += 1
            all_partial_rmsds.append(best_rmsd_total)
            all_match_coverage.append(best_coverage * 100.0)
        else:
            stats["failed_count"] += 1

        pos_pred_str = ""
        if best_conf_id is not None and gen_mols:
            pos_pred_str = mol_to_coords_str(gen_mols[best_conf_id])

        csv_rows.append(
            {
                "Index": i,
                "True SMILES": true_std,
                "Predicted SMILES": pred_std,
                "SMILES Correct": smi_correct,
                "Formula Match": formula_match,
                "Heavy Atom Match": heavy_atom_match,
                "Total Atoms": n_true_atoms if n_true_atoms > 0 else "",
                "MCS Atoms": best_mcs_atoms if is_success else "",
                "MCS Coverage (%)": mcs_pct,
                "Match Coverage (%)": match_pct,
                "Extra Atoms (Hungarian)": best_extra if is_success else "",
                "RMSD Total (Å)": f"{best_rmsd_total:.4f}" if best_rmsd_total is not None else "",
                "RMSD MCS Only (Å)": f"{best_rmsd_mcs:.4f}" if best_rmsd_mcs is not None else "",
                "RMSD < 0.5": (best_rmsd_total < 0.5) if (best_rmsd_total is not None) else False,
                "RMSD < 1.0": (best_rmsd_total < 1.0) if (best_rmsd_total is not None) else False,
                "RMSD < 2.0": (best_rmsd_total < 2.0) if (best_rmsd_total is not None) else False,
                "Best Conf ID": best_conf_id if best_conf_id is not None else "",
                "True Conformation": pos_true_str,
                "Predicted Conformation": pos_pred_str,
                "Status": status,
                "Heavy Only": heavy_only,
            }
        )

    df = pd.DataFrame(csv_rows)

    print(f"\nWriting file: {args.output_csv}")
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("📊 Stats Summary (MCS + Hungarian Method)")
    print("=" * 70)

    print(f"Calculation mode: {'All atoms (with H)' if args.use_all_atoms else 'Heavy atoms only'}")
    print(
        f"Final alignment: {'Full pair set P (Kabsch)' if args.final_align_full else 'MCS core frame (methodology default)'}"
    )
    print(f"Atom count mismatch: {'Forced in (ok_partial)' if args.allow_atom_mismatch else 'Discarded (default)'}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total conformations: {stats['total_confs']}")
    print("-" * 70)
    print(
        f"RMSD computed successfully (strict, whole molecule): {stats['valid_rmsd_count']} "
        f"({stats['valid_rmsd_count']/stats['total_samples']*100:.2f}% of samples)"
    )
    if args.allow_atom_mismatch:
        print(f"Partial RMSD (different atom counts, ok_partial): {stats['partial_rmsd_count']}")
    print(f"Atom count mismatch (not counted / discarded): {stats['atom_count_mismatch']}")
    print(f"Failed: {stats['failed_count']}")

    if all_mcs_coverage:
        print(f"\nMCS Coverage Stats (n={len(all_mcs_coverage)}):")
        print(f"  - Mean MCS coverage: {np.mean(all_mcs_coverage):.2f}%")
        print(f"  - Median MCS coverage: {np.median(all_mcs_coverage):.2f}%")
        print(f"  - Min MCS coverage:  {min(all_mcs_coverage):.2f}%")
        print(f"  - Max MCS coverage: {max(all_mcs_coverage):.2f}%")

        print("\nHungarian Extra-Atoms Stats:")
        print(f"  - Mean extra atoms: {np.mean(all_extra_atoms):.2f}")
        print(f"  - Median extra atoms: {np.median(all_extra_atoms):.2f}")
        print(f"  - Max extra atoms: {max(all_extra_atoms)}")

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
        print("\n⚠️ No valid RMSD data could be computed")

    if all_partial_rmsds:
        print("\n" + "-" * 70)
        print("[Partial RMSD Stats — different atom counts, ok_partial, diagnostic only]")
        print("-" * 70)
        print(f"Samples:              {len(all_partial_rmsds)}")
        print(f"Mean partial RMSD:    {np.mean(all_partial_rmsds):.4f} Å")
        print(f"Median partial RMSD:  {np.median(all_partial_rmsds):.4f} Å")
        print(f"Mean match coverage:  {np.mean(all_match_coverage):.2f}%")
        print(f"Median match coverage:{np.median(all_match_coverage):.2f}%")
        print(f"Min match coverage:   {min(all_match_coverage):.2f}%")
        print(
            "Note: the lower the coverage, the less trustworthy this partial RMSD is — never merge it into the strict whole-molecule RMSD."
        )

    if args.allow_atom_mismatch and all_partial_rmsds:
        merged = all_valid_rmsds + all_partial_rmsds
        print("\n" + "-" * 70)
        print("[Overall RMSD Stats after merging partial matches — strict + ok_partial]")
        print("-" * 70)
        print(f"Samples:              {len(merged)} (strict {len(all_valid_rmsds)} + partial {len(all_partial_rmsds)})")
        print(f"Share of samples covered: {len(merged)/stats['total_samples']*100:.2f}%")
        print(f"Mean RMSD:            {np.mean(merged):.4f} Å")
        print(f"Median RMSD:           {np.median(merged):.4f} Å")
        print(f"RMSD < 0.5 Å:        {np.mean([r < 0.5 for r in merged]) * 100:.2f}%")
        print(f"RMSD < 1.0 Å:        {np.mean([r < 1.0 for r in merged]) * 100:.2f}%")
        print(f"RMSD < 2.0 Å:        {np.mean([r < 2.0 for r in merged]) * 100:.2f}%")
        print(f"Min RMSD:           {min(merged):.4f} Å")
        print(f"Max RMSD:           {max(merged):.4f} Å")

    print("=" * 70)


if __name__ == "__main__":
    main()
