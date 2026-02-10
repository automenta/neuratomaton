import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .models import ANAModel, BaselineSSM
from .data import AssociativeRecallDataset, TextDataset
from .config import ANAConfig, TrainingConfig, DataConfig
from .eval import CopyTaskDataset, ReverseTaskDataset, run_eval_task
import os
import json
import numpy as np
import time

def train_one_epoch(model, dataloader, optimizer, criterion, device, writer, global_step, force_prob=0.0, log_interval=10):
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
        
        return_info = (batch_idx % log_interval == 0)
        
        if isinstance(model, BaselineSSM):
            logits, info_log = model(x)
        else:
            logits, info_log = model(x, return_info=return_info, force_prob=force_prob)
        
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
        
        if batch_idx % log_interval == 0:
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            if info_log and len(info_log) > 0:
                first_stats = info_log[0]
                for k, v in first_stats.items():
                    writer.add_scalar(f'Train/{k}', v, global_step)
        
        global_step += 1
    
    return total_loss / len(dataloader), global_step

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    needle_correct = 0
    needle_samples = 0
    
    stats = {'ga_0': [], 'ret_gate': [], 'thinking_steps': []}
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(device)
            else:
                x, y = batch
                mask = None
            
            x, y = x.to(device), y.to(device)
            
            if isinstance(model, BaselineSSM):
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
            
            preds = torch.argmax(logits, dim=-1)
            last_pred = preds[:, -1]
            last_target = y[:, -1]
            needle_correct += (last_pred == last_target).float().sum().item()
            needle_samples += x.size(0)
            
            total_correct += (preds == y).float().sum().item()
            total_samples += y.numel()
            
            for info in info_log:
                if 'ga_0' in info: stats['ga_0'].append(info['ga_0'])
                if 'ret_gate' in info: stats['ret_gate'].append(info['ret_gate'])
                if 'thinking_steps' in info: stats['thinking_steps'].append(info['thinking_steps'])
    
    avg_stats = {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    needle_acc = needle_correct / needle_samples if needle_samples > 0 else 0.0
    ppl = np.exp(total_loss / len(dataloader))
    
    return total_loss / len(dataloader), avg_stats, acc, needle_acc, ppl

def col_fn(batch):
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

def get_device(device_str):
    if device_str == "auto":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def run_training(ana_config: ANAConfig, train_config: TrainingConfig, data_config: DataConfig, model_type="ana"):
    device = get_device(train_config.device)
    print(f"Using device: {device}")
    
    log_dir = f"archive/logs/run_{train_config.stage}_{model_type}_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")
    
    if train_config.stage in ['2a', '3a']:
        min_noise = data_config.min_noise
        max_noise = data_config.max_noise
        if train_config.complexity_curriculum:
            min_noise = 1
            max_noise = 5
        
        dataset = AssociativeRecallDataset(
            size=data_config.dataset_size,
            vocab_size=data_config.vocab_size,
            min_noise=min_noise,
            max_noise=max_noise
        )
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=col_fn, num_workers=4, pin_memory=True)
        ana_config.vocab_size = data_config.vocab_size
    
    elif train_config.stage == '2b':
        os.makedirs('data', exist_ok=True)
        if not os.path.exists(data_config.dataset_path):
            os.system(f"cat ana/*.py README.md > {data_config.dataset_path}")
        
        dataset = TextDataset(data_config.dataset_path, seq_len=data_config.seq_len)
        if len(dataset) == 0:
            print("Error: Corpus empty.")
            return
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        ana_config.vocab_size = 256
    
    if model_type == "baseline":
        print("Initializing BaselineSSM...")
        model = BaselineSSM(ana_config).to(device)
    else:
        print("Initializing ANAModel...")
        model = ANAModel(ana_config).to(device)
    
    if train_config.stage == '2b' and model_type != "baseline":
        checkpoint_path = f"{train_config.output_dir}/model_stage2a_{model_type}.pt"
        if os.path.exists(checkpoint_path):
            print(f"Loading Stage 2A weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} parameters from checkpoint")
        
        if ana_config.use_hololink:
            print("Freezing HoloLink parameters...")
            for layer in model.layers:
                if 'holo' in layer:
                    for param in layer['holo'].parameters():
                        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.learning_rate)
    
    print(f"Starting Stage {train_config.stage} Training...")
    history = {
        'loss': [], 'ppl': [], 'acc': [], 'needle_acc': [],
        'ga_0': [], 'ret_gate': [], 'force_prob': [], 'thinking_steps': [],
        'copy_acc': [], 'reverse_acc': []
    }
    
    global_step = 0
    
    for epoch in range(train_config.epochs):
        force_prob = 0.0
        
        if train_config.stage == '3a':
            curr_epochs = train_config.curriculum_epochs
            start_prob = train_config.start_force_prob
            
            if epoch < curr_epochs:
                force_prob = start_prob * (1.0 - (epoch / float(curr_epochs)))
            else:
                force_prob = 0.0
            
            if train_config.complexity_curriculum and epoch > 2:
                max_noise = min(data_config.max_noise, 5 + (epoch - 2) * 5)
                dataset = AssociativeRecallDataset(
                    size=data_config.dataset_size,
                    vocab_size=data_config.vocab_size,
                    min_noise=data_config.min_noise,
                    max_noise=max_noise
                )
                dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=col_fn)
        
        train_loss, global_step = train_one_epoch(
            model, dataloader, optimizer, criterion, device, writer, global_step,
            force_prob=force_prob, log_interval=train_config.log_interval
        )
        
        val_loss, stats, acc, needle_acc, ppl = evaluate(model, dataloader, criterion, device)
        
        copy_score = 0.0
        rev_score = 0.0
        if train_config.stage == '3a' and epoch % 2 == 0:
            copy_ds = CopyTaskDataset(size=200, vocab_size=ana_config.vocab_size, seq_len=10)
            rev_ds = ReverseTaskDataset(size=200, vocab_size=ana_config.vocab_size, seq_len=10)
            copy_score = run_eval_task(model, copy_ds, device)
            rev_score = run_eval_task(model, rev_ds, device)
        
        print(f"Epoch {epoch+1} | Force: {force_prob:.2f} | Loss: {val_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.4f} | Needle: {needle_acc:.4f} | Gates: A={stats['ga_0']:.2f}, Ret={stats['ret_gate']:.2f} | Copy: {copy_score:.2f} | Rev: {rev_score:.2f}")
        
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Perplexity', ppl, epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Needle_Accuracy', needle_acc, epoch)
        writer.add_scalar('Train/Force_Prob', force_prob, epoch)
        writer.add_scalar('Val/Gate_A_Mean', stats['ga_0'], epoch)
        writer.add_scalar('Val/Gate_Ret_Mean', stats['ret_gate'], epoch)
        
        if copy_score > 0:
            writer.add_scalar('Eval/Copy_Acc', copy_score, epoch)
            writer.add_scalar('Eval/Reverse_Acc', rev_score, epoch)
        
        history['loss'].append(val_loss)
        history['ppl'].append(ppl)
        history['acc'].append(acc)
        history['needle_acc'].append(needle_acc)
        history['force_prob'].append(force_prob)
        history['ga_0'].append(stats['ga_0'])
        history['ret_gate'].append(stats['ret_gate'])
        history['thinking_steps'].append(stats.get('thinking_steps', 0))
        history['copy_acc'].append(copy_score)
        history['reverse_acc'].append(rev_score)
    
    os.makedirs(train_config.output_dir, exist_ok=True)
    
    results_path = f'{train_config.output_dir}/results_stage{train_config.stage}_{model_type}.json'
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    if train_config.save_checkpoints:
        model_path = f'{train_config.output_dir}/model_stage{train_config.stage}_{model_type}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    writer.close()
    print(f"Stage {train_config.stage} Complete. Results saved to {results_path}")
    return history

def main():
    ana_config = ANAConfig()
    train_config = TrainingConfig()
    data_config = DataConfig()
    
    import sys
    if len(sys.argv) > 1:
        train_config.stage = sys.argv[1]
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    else:
        model_type = "ana"
    
    run_training(ana_config, train_config, data_config, model_type)

if __name__ == '__main__':
    main()
