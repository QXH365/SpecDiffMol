# train.py

import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.misc import *
from utils.common import get_optimizer, get_scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file (e.g., configs/train.yml)")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, required=True)
    args = parser.parse_args()

    # --- 1. Config and Log Setup ---
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config_name = os.path.basename(config_path).split('.')[0]
    seed_all(config.train.seed)

    if resume:
        log_dir = resume_from
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info("Mode: Training full model from scratch.")
    logger.info(args)
    logger.info(config)

    # --- 2. Dataset Loading ---
    logger.info('Loading spectral datasets...')
    train_set = ConformationDataset(config.dataset.train)
    val_set = ConformationDataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_set, 
        config.train.batch_size, 
        shuffle=True
    ))
    val_loader = DataLoader(
        val_set, 
        config.train.batch_size, 
        shuffle=False
    )
    logger.info(f'Dataset loaded: Train {len(train_set)} | Val {len(val_set)}')

    # --- 3. Model, Optimizer, Scheduler ---
    logger.info('Building model...')
    # Fixed training_phase='finetune' to build full model with spectral module
    model = get_model(config, training_phase='finetune').to(args.device)
    
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1
    
    # --- 4. Resume from Checkpoint ---
    if resume and args.resume_iter is not None:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # --- 5. Train & Validation Loop ---
    def train(it):
        model.train()
        optimizer.zero_grad()
        try:
            batch = next(train_iterator).to(args.device)
        except StopIteration:
            # Re-init iterator if exhausted (though inf_iterator handles this usually)
             return

        loss, loss_global, loss_local = model.get_loss(batch=batch, anneal_power=config.train.anneal_power)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        
        logger.info('[Train] Iter %05d | Total Loss %.3f | Global Loss %.3f | Local Loss %.3f | Grad Norm %.2f | LR %.6f' % (
            it, loss.item(), loss_global.item(), loss_local.item(), orig_grad_norm, optimizer.param_groups[0]['lr'],
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_global', loss_global, it)
        writer.add_scalar('train/loss_local', loss_local, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()


    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validation'):
                batch = batch.to(args.device)
                loss, _, _ = model.get_loss(
                    batch=batch, anneal_power=config.train.anneal_power,
                )
                sum_loss += loss.item() * batch.num_graphs
                sum_n += batch.num_graphs
        avg_loss = sum_loss / sum_n if sum_n > 0 else 0

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        logger.info('[Val] Iter %05d | Avg Loss %.3f' % (it, avg_loss))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        
        return avg_loss

    # --- 6. Main Loop ---
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating training...')