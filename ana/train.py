
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models import ANAModel
from .data import AssociativeRecallDataset
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
                stats['ga_A'].append(info['ga_A'])
                stats['ga_B'].append(info['ga_B'])
                stats['ret_gate'].append(info['ret_gate'])
                
    # Mean stats
    avg_stats = {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
    acc = total_correct / total_samples
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    VOCAB_SIZE = 40
    D_MODEL = 64
    STATE_DIM = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 16
    EPOCHS = 6 # Stage 2A
    
    # Dataset
    # Noise range 20 to 50
    dataset = AssociativeRecallDataset(size=2000, vocab_size=VOCAB_SIZE, min_noise=20, max_noise=50)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=col_fn)
    
    # Model: Single Phase 2 ANA
    model = ANAModel(VOCAB_SIZE, D_MODEL, STATE_DIM, NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starting Stage 2A: Associative Recall Training...")
    
    history = {'loss': [], 'ga_A': [], 'ga_B': [], 'ret': []}
    
    for epoch in range(EPOCHS):
        val_loss, stats = evaluate(model, dataloader, criterion, device) # Pre-eval
        print(f"Epoch {epoch} (Pre) | Loss: {val_loss:.4f} | Gates: A={stats['ga_A']:.2f}, B={stats['ga_B']:.2f}, Ret={stats['ret_gate']:.2f}")
        
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        val_loss, stats = evaluate(model, dataloader, criterion, device)
        
        print(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | Gates: A={stats['ga_A']:.2f}, B={stats['ga_B']:.2f}, Ret={stats['ret_gate']:.2f}")
        
        history['loss'].append(val_loss)
        history['ga_A'].append(stats['ga_A'])
        history['ga_B'].append(stats['ga_B'])
        history['ret'].append(stats['ret_gate'])
        
    # Save
    with open('archive/results/results_phase2a.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    # Save Model Weights
    torch.save(model.state_dict(), 'archive/results/model_stage2a.pt')
        
    print("Stage 2A Complete.")
    
def train_stage_2b():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    VOCAB_SIZE_TXT = 256 # Char level
    D_MODEL = 64
    STATE_DIM = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 4 # Warmup
    
    # Dataset (Use local files as fallback)
    # Concatenate some files
    os.system("cat ana/*.py README.md > data/corpus.txt")
    dataset = TextDataset('data/corpus.txt', seq_len=64)
    if len(dataset) == 0:
        print("Error: Corpus empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model: New vocab size
    model = ANAModel(VOCAB_SIZE_TXT, D_MODEL, STATE_DIM, NUM_LAYERS).to(device)
    
    # Load Weights from Stage 2A (Partial)
    if os.path.exists('model_stage2a.pt'):
        print("Loading Stage 2A weights (Filtering embedding/head)...")
        state_dict = torch.load('model_stage2a.pt')
        model_dict = model.state_dict()
        
        # Filter out mismatching keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("Warning: Stage 2A weights not found. Starting fresh.")
        
    # Freeze HoloLink? "Train only Tracks and Controller"
    # To freeze HoloLink:
    for layer in model.layers:
        for param in layer['holo'].parameters():
            param.requires_grad = False
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    print("Starting Stage 2B: Text Warmup (Holo Frozen)...")
    history = {'loss': []}
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f}")
        history['loss'].append(train_loss)
        
    with open('results_phase2b.json', 'w') as f:
        json.dump(history, f, indent=2)

def train_stage_3a():
    """
    Stage 3A: Associative Recall with Forced Curriculum
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    VOCAB_SIZE = 40
    D_MODEL = 64
    STATE_DIM = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 16
    EPOCHS = 10 # More epochs to allow annealing
    
    # Dataset
    # DEBUG: Short noise to verify mechanism
    dataset = AssociativeRecallDataset(size=2000, vocab_size=VOCAB_SIZE, min_noise=1, max_noise=5)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=col_fn)
    
    # Model
    model = ANAModel(VOCAB_SIZE, D_MODEL, STATE_DIM, NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starting Stage 3A: Forced Holo-Link Curriculum...")
    history = {'loss': [], 'force_prob': []}
    
    for epoch in range(EPOCHS):
        # Curriculum:
        # Epoch 0-2: Force 100%
        # Epoch 3-5: Anneal 1.0 -> 0.0
        # Epoch 6+: 0%
        if epoch < 3:
            force_prob = 1.0
        elif epoch < 6:
            force_prob = 1.0 - ((epoch - 2) / 3.0) # 0.66, 0.33, 0.0
        else:
            force_prob = 0.0
            
        # Training
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, force_prob=force_prob)
        
        # Eval (without forcing to see if it learned)
        val_loss, stats, acc = evaluate(model, dataloader, criterion, device)
        
        print(f"Epoch {epoch+1} | Force: {force_prob:.2f} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | Gates: A={stats['ga_A']:.2f}, Ret={stats['ret_gate']:.2f}")
        
        history['loss'].append(val_loss)
        history['force_prob'].append(force_prob)

    with open('results_phase3a.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    # Toggle stages
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '2b':
            from data import TextDataset
            train_stage_2b()
        elif sys.argv[1] == '3a':
            train_stage_3a()
    else:
        main()
