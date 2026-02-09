
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models import ANAModel, BaselineSSM
from .data import AssociativeRecallDataset, TextDataset
from .config import ANAConfig, TrainingConfig, DataConfig
import matplotlib.pyplot as plt
import os
import json
import numpy as np

def train_one_epoch(model, dataloader, optimizer, criterion, device, force_prob=0.0):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(device)
        else:
            x, y = batch
            mask = None
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Check if model supports force_prob
        # BaselineSSM doesn't have force_prob arg in forward
        if isinstance(model, BaselineSSM):
            logits, _ = model(x)
        else:
            logits, _ = model(x, force_prob=force_prob)
        
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
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    stats = {
        'ga_A': [], 'ga_B': [], 'ret_gate': []
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
            
            # BaselineSSM returns empty info_log
            if isinstance(model, BaselineSSM):
                 logits, info_log = model(x), []
                 # Wait, Baseline forward returns (logits, [])
                 logits, info_log = model(x)
            else:
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
                if 'ga_A' in info: stats['ga_A'].append(info['ga_A'])
                if 'ga_B' in info: stats['ga_B'].append(info['ga_B'])
                if 'ret_gate' in info: stats['ret_gate'].append(info['ret_gate'])
                
    # Mean stats
    avg_stats = {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
    if total_samples > 0:
        acc = total_correct / total_samples
    else:
        acc = 0.0
    return total_loss / len(dataloader), avg_stats, acc

def col_fn(batch):
    # Padding collision
    # Check if we have mask
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

def run_training(ana_config: ANAConfig, train_config: TrainingConfig, data_config: DataConfig, model_type="ana"):
    device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if train_config.stage in ['2a', '3a']:
        dataset = AssociativeRecallDataset(
            size=data_config.dataset_size,
            vocab_size=data_config.vocab_size,
            min_noise=data_config.min_noise,
            max_noise=data_config.max_noise
        )
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=col_fn)
        # Update model vocab size to match dataset
        ana_config.vocab_size = data_config.vocab_size
    elif train_config.stage == '2b':
         # Concatenate some files if corpus not present
        if not os.path.exists('data/corpus.txt'):
             if not os.path.exists('data'): os.makedirs('data')
             os.system("cat ana/*.py README.md > data/corpus.txt")

        dataset_path = data_config.dataset_path if data_config.dataset_path else 'data/corpus.txt'
        dataset = TextDataset(dataset_path, seq_len=data_config.seq_len)
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
        # Update model vocab size for text
        ana_config.vocab_size = 256

    if model_type == "baseline":
        print("Initializing BaselineSSM...")
        model = BaselineSSM(ana_config).to(device)
    else:
        print("Initializing ANAModel...")
        model = ANAModel(ana_config).to(device)
    
    # Load weights if needed (e.g. for 2b or 3a continuation)
    # Simple logic: if model file exists from previous stage, try load?
    # Keeping simple for now.

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    
    print(f"Starting Stage {train_config.stage} Training...")
    history = {'loss': [], 'ga_A': [], 'ga_B': [], 'ret': [], 'force_prob': []}
    
    for epoch in range(train_config.epochs):
        force_prob = 0.0
        if train_config.stage == '3a':
             # Curriculum logic
            if epoch < 3:
                force_prob = 1.0
            elif epoch < 6:
                force_prob = 1.0 - ((epoch - 2) / 3.0)
            else:
                force_prob = 0.0

        # Train
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, force_prob=force_prob)
        
        # Eval
        val_loss, stats, acc = evaluate(model, dataloader, criterion, device)

        print(f"Epoch {epoch+1} | Force: {force_prob:.2f} | Loss: {val_loss:.4f} | Acc: {acc:.4f} | Gates: A={stats['ga_A']:.2f}, Ret={stats['ret_gate']:.2f}")
        
        history['loss'].append(val_loss)
        history['force_prob'].append(force_prob)
        history['ga_A'].append(stats['ga_A'])
        history['ga_B'].append(stats['ga_B'])
        history['ret'].append(stats['ret_gate'])
        
    # Save
    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir)
        
    with open(f'{train_config.output_dir}/results_stage{train_config.stage}_{model_type}.json', 'w') as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), f'{train_config.output_dir}/model_stage{train_config.stage}_{model_type}.pt')
    print(f"Stage {train_config.stage} Complete.")

def main():
    # Example Usage
    ana_config = ANAConfig()
    train_config = TrainingConfig()
    data_config = DataConfig()
    
    # Simple CLI dispatch
    import sys
    if len(sys.argv) > 1:
        train_config.stage = sys.argv[1]
    
    run_training(ana_config, train_config, data_config)

if __name__ == '__main__':
    main()
