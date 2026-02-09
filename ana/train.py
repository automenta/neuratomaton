
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .models import ANAModel
from .data import AssociativeRecallDataset, TextDataset
from .config import ANAConfig
from .eval import CopyTaskDataset, ReverseTaskDataset, run_eval_task
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import time

def train_one_epoch(model, dataloader, optimizer, criterion, device, writer, global_step, force_prob=0.0):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(device)
        else:
            x, y = batch
            mask = None
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Get info for logging
        logits, info_log = model(x, return_info=(batch_idx % 10 == 0), force_prob=force_prob)
        
        loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_raw = loss_raw.view(y.size())
        
        if mask is not None:
            loss = (loss_raw * mask).sum() / mask.sum()
        else:
            loss = loss_raw.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Step Logging
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            # Log gates
            if info_log:
                # info_log is list of dicts (one per layer, usually layer 0 is logged)
                # ANAModel returns list of dicts.
                if len(info_log) > 0:
                    first_stats = info_log[0]
                    for k, v in first_stats.items():
                        writer.add_scalar(f'Train/Gate_{k}', v, global_step)

        global_step += 1

    return total_loss / len(dataloader), global_step

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    stats = {
        'ga_0': [], 'ret_gate': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(device)
            else:
                x, y = batch
                mask = None
                
            x, y = x.to(device), y.to(device)
            
            logits, info_log = model(x, return_info=True)
            
            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_raw = loss_raw.view(y.size())
            
            if mask is not None:
                loss = (loss_raw * mask).sum() / mask.sum()
            else:
                loss = loss_raw.mean()
                
            total_loss += loss.item()
            
            # Retrieval Accuracy (Last Token)
            last_logits = logits[:, -1, :] 
            last_targets = y[:, -1]        
            preds = torch.argmax(last_logits, dim=-1)
            correct = (preds == last_targets).float().sum()
            total_correct += correct.item()
            total_samples += x.size(0)
            
            # Aggregate stats
            for info in info_log:
                if 'ga_0' in info: stats['ga_0'].append(info['ga_0'])
                if 'ret_gate' in info: stats['ret_gate'].append(info['ret_gate'])
                
    # Mean stats
    avg_stats = {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
    acc = total_correct / total_samples
    return total_loss / len(dataloader), avg_stats, acc

def col_fn(batch):
    # Padding collision
    has_mask = len(batch[0]) == 3
    max_len = max([item[0].size(0) for item in batch])
    
    xs, ys, ms = [], [], []
    for item in batch:
        if has_mask:
            x, y, mask = item
        else:
            x, y = item
            mask = None
            
        pad = max_len - x.size(0)
        if pad > 0:
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])
            if mask is not None:
                 mask = torch.cat([mask, torch.zeros(pad, dtype=torch.float)])
                 
        xs.append(x)
        ys.append(y)
        if mask is not None:
            ms.append(mask)
        
    if has_mask:
        return torch.stack(xs), torch.stack(ys), torch.stack(ms)
    else:
        return torch.stack(xs), torch.stack(ys)

def train_stage_2a(config: ANAConfig):
    print(f"Using device: {config.device}")
    
    # Logging
    log_dir = f"archive/logs/run_2a_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # Dataset
    dataset = AssociativeRecallDataset(size=2000, vocab_size=config.vocab_size, min_noise=20, max_noise=50)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=col_fn)
    
    # Model
    model = ANAModel(config).to(config.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print("Starting Stage 2A: Associative Recall Training...")
    
    history = {'loss': [], 'ga_0': [], 'ret': []}
    global_step = 0
    
    for epoch in range(config.epochs):
        train_loss, global_step = train_one_epoch(model, dataloader, optimizer, criterion, config.device, writer, global_step)
        val_loss, stats, acc = evaluate(model, dataloader, criterion, config.device)
        
        print(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | Acc: {acc:.4f} | Gates: A={stats['ga_0']:.2f}, Ret={stats['ret_gate']:.2f}")

        # Epoch Logging
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Gate_A_Mean', stats['ga_0'], epoch)
        writer.add_scalar('Val/Gate_Ret_Mean', stats['ret_gate'], epoch)
        
        history['loss'].append(val_loss)
        history['ga_0'].append(stats['ga_0'])
        history['ret'].append(stats['ret_gate'])
        
    # Save
    os.makedirs('archive/results', exist_ok=True)
    with open('archive/results/results_phase2a.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    torch.save(model.state_dict(), 'archive/results/model_stage2a.pt')
    writer.close()
    print("Stage 2A Complete.")
    
def train_stage_2b(config: ANAConfig):
    print(f"Using device: {config.device}")

    log_dir = f"archive/logs/run_2b_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")
    
    config.vocab_size = 256
    
    os.makedirs('data', exist_ok=True)
    os.system("cat ana/*.py README.md > data/corpus.txt")
    dataset = TextDataset('data/corpus.txt', seq_len=64)
    if len(dataset) == 0:
        print("Error: Corpus empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = ANAModel(config).to(config.device)
    
    if os.path.exists('archive/results/model_stage2a.pt'):
        print("Loading Stage 2A weights...")
        state_dict = torch.load('archive/results/model_stage2a.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("Warning: Stage 2A weights not found.")
        
    if config.use_hololink:
        for layer in model.layers:
            if 'holo' in layer:
                for param in layer['holo'].parameters():
                    param.requires_grad = False
            
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    
    print("Starting Stage 2B: Text Warmup...")
    global_step = 0
    
    for epoch in range(config.epochs):
        train_loss, global_step = train_one_epoch(model, dataloader, optimizer, criterion, config.device, writer, global_step)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f}")
        writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
        
    writer.close()

def train_stage_3a(config: ANAConfig):
    print(f"Using device: {config.device}")
    
    log_dir = f"archive/logs/run_3a_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # Initial difficulty
    min_noise = 1
    max_noise = 5
    
    model = ANAModel(config).to(config.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print("Starting Stage 3A: Forced Curriculum...")
    global_step = 0
    
    for epoch in range(config.epochs):
        # Complexity Curriculum: Increase noise every few epochs
        if epoch > 2:
            max_noise = min(50, 5 + (epoch - 2) * 5)

        dataset = AssociativeRecallDataset(size=2000, vocab_size=config.vocab_size, min_noise=min_noise, max_noise=max_noise)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=col_fn)

        # Forcing Curriculum
        if epoch < 3:
            force_prob = 1.0
        elif epoch < 6:
            force_prob = 1.0 - ((epoch - 2) / 3.0)
        else:
            force_prob = 0.0
            
        train_loss, global_step = train_one_epoch(model, dataloader, optimizer, criterion, config.device, writer, global_step, force_prob=force_prob)
        
        val_loss, stats, acc = evaluate(model, dataloader, criterion, config.device)
        
        # Reasoning Evals
        copy_score = 0.0
        rev_score = 0.0
        if epoch % 2 == 0:
            copy_ds = CopyTaskDataset(size=200, vocab_size=config.vocab_size, seq_len=10)
            rev_ds = ReverseTaskDataset(size=200, vocab_size=config.vocab_size, seq_len=10)
            copy_score = run_eval_task(model, copy_ds, config.device)
            rev_score = run_eval_task(model, rev_ds, config.device)
        
        print(f"Epoch {epoch+1} | Force: {force_prob:.2f} | Noise: {max_noise} | Loss: {val_loss:.4f} | Acc: {acc:.4f} | Copy: {copy_score:.2f} | Rev: {rev_score:.2f}")

        writer.add_scalar('Train/Max_Noise', max_noise, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Train/Force_Prob', force_prob, epoch)
        writer.add_scalar('Eval/Copy_Acc', copy_score, epoch)
        writer.add_scalar('Eval/Rev_Acc', rev_score, epoch)

    writer.close()

if __name__ == '__main__':
    import sys
    config = ANAConfig()
    if len(sys.argv) > 1:
        # CLI usage (legacy)
        pass
    else:
        train_stage_2a(config)
