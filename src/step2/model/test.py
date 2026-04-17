import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from rdkit import Chem

from models.epsnet import get_model
from utils.misc import seed_all, get_logger, get_new_log_dir
from utils.chem import set_rdmol_positions 

def main():
    parser = argparse.ArgumentParser(description='End-to-end inference: using unified molecular templates')
    parser.add_argument('ckpt', type=str, help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, default=None, help='Path to the config file (optional)')
    parser.add_argument('--data_path', type=str, default='../preprocess/qme14s_custom_split_without_fzs/test.pkl', help='Path to the test dataset .pkl file')
    parser.add_argument('--tag', type=str, default='', help='Add a tag to the output directory')
    parser.add_argument('--num_confs', type=int, default=None, help='Number of conformations to generate per molecule')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the test set')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of the test set')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    
    parser.add_argument('--n_steps', type=int, default=100, help='Number of sampling steps')
    parser.add_argument('--w_global', type=float, default=0.5, help='Weight of global gradient')
    
    args = parser.parse_args()
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    config_path_in_ckpt_dir = glob(os.path.join(os.path.dirname(args.ckpt), '*.yml'))
    if config_path_in_ckpt_dir:
        config_path = config_path_in_ckpt_dir[0]
    elif args.config:
        config_path = args.config
    else:
        raise ValueError("Error: Configuration file not found.")

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    seed_all(config.train.seed if hasattr(config.train, 'seed') else 42)
    
    log_dir = os.path.dirname(args.ckpt)
    output_dir = get_new_log_dir(log_dir, 'inference', tag=args.tag)
    logger = get_logger('inference', output_dir)
    logger.info(f"Arguments: {args}")

    logger.info(f"Loading test dataset: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    end_idx = args.end_idx if args.end_idx is not None else len(test_data)
    test_data_selected = test_data[args.start_idx:end_idx]
    logger.info(f"Selected {len(test_data_selected)} molecules for inference")

    test_loader = DataLoader(
        test_data_selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    logger.info("Initializing model (finetune mode)...")
    model = get_model(config, training_phase='finetune').to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    saved_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating conformations"):
            try:
                original_data_list = batch.to_data_list()
                num_samples = args.num_confs

                if num_samples > 1:
                    repeated_list = [d for d in original_data_list for _ in range(num_samples)]
                    inference_batch = Batch.from_data_list(repeated_list).to(args.device)
                else:
                    inference_batch = batch.to(args.device)
                
                pos_gen_all, _ = model.langevin_dynamics_sample_ode(
                    batch=inference_batch,
                    n_steps=args.n_steps,
                    w_global=args.w_global,
                )
                pos_gen_all = pos_gen_all.cpu()
                
                num_nodes_list = [d.num_nodes for d in inference_batch.to_data_list()]
                pos_gen_list = pos_gen_all.split(num_nodes_list)

                current_gen_idx = 0
                for data in original_data_list:
                    confs_for_this_mol = pos_gen_list[current_gen_idx : current_gen_idx + num_samples]
                    current_gen_idx += num_samples
                    
                    mol_template = Chem.Mol(data.rdmol)
                    
                    mol_true = set_rdmol_positions(Chem.Mol(data.rdmol), data.pos)
                    
                    generated_mols = []
                    for pos_gen_conf in confs_for_this_mol: 
                        try:
                            mol_gen = set_rdmol_positions(Chem.Mol(data.rdmol), pos_gen_conf)
                            generated_mols.append(mol_gen)
                        except Exception: 
                            generated_mols.append(None)

                    true_smiles = getattr(data, 'true_smiles', data.smiles)
                    
                    saved_results.append({
                        'pred_smiles': data.smiles,
                        'true_smiles': true_smiles,
                        'true_mol': mol_true,
                        'gen_mols': generated_mols,
                    })

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue

    if saved_results: 
        logger.info(f"\n{'='*50}\nInference Complete\n{'='*50}")
        logger.info(f"Generated conformation data for {len(saved_results)} molecules.")
        
        results_path = os.path.join(output_dir, 'inference_results_all.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(saved_results, f)
            
        logger.info(f"Full results (including molecular objects) saved to: {results_path}")
        logger.info("Please run the evaluation script to calculate RMSD.")
        
        logger.info("\n" + "="*50)
        logger.info("📋 Important Notes:")
        logger.info("="*50)
        logger.info("1. Ground truth and generated molecules use the SAME molecular template (data.rdmol)")
        logger.info("2. This ensures atom index consistency for correct RMSD calculation")
        logger.info("3. RMSD can be calculated even if true_smiles != pred_smiles")
        logger.info("4. This method evaluates the quality of conformations generated given the predicted graph structure")
        logger.info("="*50)
    else:
        logger.error("No valid results were generated!")

if __name__ == '__main__':
    main()