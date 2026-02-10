import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ana.config_v2 import ANAv2Config, Trainingv2Config, Datav2Config
from ana.model_v3 import ANAv2Model
from ana.data import AssociativeRecallDataset, TextDataset
import os
import json
import numpy as np
import time


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


class TrainerV2:
    def __init__(self, config: ANAv2Config, train_config: Trainingv2Config, data_config: Datav2Config):
        self.config = config
        self.train_config = train_config
        self.data_config = data_config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and train_config.device == "auto" else train_config.device)
        
        self.log_dir = f"archive/logs_v2/run_{train_config.stage}_{int(time.time())}"
        self.writer = SummaryWriter(self.log_dir)
        
        self.model = ANAv2Model(config).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")
        
    def get_device(self):
        return self.device
    
    def setup_stage0_curriculum(self):
        print("Setting up Stage 0: Baseline ANA (Frozen Meta)")
        
        for name, param in self.model.named_parameters():
            if 'stack' in name.lower() or 'cortex' in name.lower():
                param.requires_grad = False
                print(f"Frozen: {name}")
        
        dataset = AssociativeRecallDataset(
            size=self.data_config.dataset_size,
            vocab_size=self.data_config.vocab_size,
            min_noise=5,
            max_noise=15
        )
        
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, 
                               shuffle=True, num_workers=0, pin_memory=True, collate_fn=col_fn)
        
        self.config.vocab_size = self.data_config.vocab_size
        self.config.use_parallel_scan = True
        
        return dataloader
    
    def setup_stage1_curriculum(self, epochs_trained=0):
        print("Setting up Stage 1: Add Stack + Gumbel Routing")
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            if 'fault_buffer' in name.lower():
                param.requires_grad = False
                print(f"Frozen: {name}")
        
        max_noise = min(10 + epochs_trained * 2, self.data_config.max_noise)
        
        dataset = AssociativeRecallDataset(
            size=self.data_config.dataset_size,
            vocab_size=self.data_config.vocab_size,
            min_noise=5,
            max_noise=int(max_noise)
        )
        
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, 
                               shuffle=True, num_workers=0, pin_memory=True, collate_fn=col_fn)
        
        return dataloader
    
    def setup_stage2_curriculum(self):
        print("Setting up Stage 2: Full Meta-Loss + Fault Traces")
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        
        corpus_path = self.data_config.dataset_path
        if corpus_path is None:
            os.makedirs('data', exist_ok=True)
            corpus_path = 'data/corpus_v2.txt'
            if not os.path.exists(corpus_path):
                os.system("cat ana/*.py README.md > " + corpus_path)
        
        if os.path.exists(corpus_path):
            text_ds = TextDataset(corpus_path, seq_len=self.data_config.seq_len)
            if len(text_ds) > 0:
                print(f"Using text corpus with {len(text_ds)} samples")
                dataloader = DataLoader(text_ds, batch_size=self.train_config.batch_size, 
                                       shuffle=True, num_workers=0, pin_memory=True)
                return dataloader
        
        max_noise = self.data_config.max_noise
        dataset = AssociativeRecallDataset(
            size=self.data_config.dataset_size,
            vocab_size=self.data_config.vocab_size,
            min_noise=20,
            max_noise=max_noise
        )
        
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, 
                               shuffle=True, num_workers=0, pin_memory=True, collate_fn=col_fn)
        
        return dataloader
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch, stage):
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_rule = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(self.device)
            else:
                x, y = batch
                mask = None
            
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            if self.config.use_parallel_scan and stage != '2':
                logits, rule_logits = self.model.forward_parallel(x)
            else:
                logits, rule_logits, info = self.model(x, return_info=True)
            
            loss_dict = self.model.compute_loss(logits, rule_logits, y)
            loss = loss_dict['total']
            
            if mask is not None:
                ce_per_pos = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction='none')
                ce_per_pos = ce_per_pos.view(y.size())
                loss = (ce_per_pos * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_dict['ce'].item()
            total_rule += loss_dict['rule'].item()
            
            if batch_idx % self.train_config.log_interval == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Train/Loss_Step', loss.item(), step)
                self.writer.add_scalar('Train/CE_Loss', loss_dict['ce'].item(), step)
                self.writer.add_scalar('Train/Rule_Loss', loss_dict['rule'].item(), step)
        
        avg_loss = total_loss / len(dataloader)
        avg_ce = total_ce / len(dataloader)
        avg_rule = total_rule / len(dataloader)
        
        return avg_loss, avg_ce, avg_rule
    
    @torch.no_grad()
    def evaluate(self, dataloader, criterion, stage):
        self.model.eval()
        total_loss = 0
        total_ce = 0
        needle_correct = 0
        needle_total = 0
        all_correct = 0
        all_total = 0
        
        stack_depths = []
        
        for batch in dataloader:
            if len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(self.device)
            else:
                x, y = batch
                mask = None
            
            x, y = x.to(self.device), y.to(self.device)
            
            info = None
            if self.config.use_parallel_scan and stage != '2':
                logits, rule_logits = self.model.forward_parallel(x)
            else:
                logits, rule_logits, info = self.model(x, return_info=True)
            
            if mask is not None:
                ce_per_pos = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction='none')
                ce_per_pos = ce_per_pos.view(y.size())
                loss = (ce_per_pos * mask).sum() / mask.sum()
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            
            total_loss += loss.item()
            total_ce += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0).item()
            
            preds = torch.argmax(logits, dim=-1)
            
            if mask is not None:
                last_pred = preds[:, -1]
                last_target = y[:, -1]
                needle_correct += (last_pred == last_target).float().sum().item()
                needle_total += x.size(0)
            
            all_correct += (preds == y).float().sum().item()
            all_total += y.numel()
            
            if info is not None:
                for i in info:
                    stack_depths.append(i['stack_depth'])
        
        avg_loss = total_loss / len(dataloader)
        avg_ce = total_ce / len(dataloader)
        ppl = np.exp(avg_ce)
        acc = all_correct / all_total if all_total > 0 else 0
        needle_acc = needle_correct / needle_total if needle_total > 0 else 0
        avg_stack_depth = np.mean(stack_depths) if stack_depths else 0
        
        return avg_loss, avg_ce, ppl, acc, needle_acc, avg_stack_depth
    
    def run_stage0(self):
        print("\n" + "="*60)
        print("STAGE 0: Baseline ANA (Frozen Meta)")
        print("="*60)
        
        dataloader = self.setup_stage0_curriculum()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=self.train_config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        history = {
            'loss': [], 'ce': [], 'ppl': [], 'acc': [], 'needle_acc': [],
            'stack_depth': [], 'rule_loss': []
        }
        
        for epoch in range(self.train_config.epochs):
            train_loss, train_ce, train_rule = self.train_epoch(dataloader, optimizer, criterion, epoch, '0')
            val_loss, val_ce, val_ppl, val_acc, needle_acc, stack_depth = self.evaluate(dataloader, criterion, '0')
            
            print(f"Epoch {epoch+1}/{self.train_config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"Needle Acc: {needle_acc:.4f} | Stack: {stack_depth:.2f}")
            
            history['loss'].append(val_loss)
            history['ce'].append(val_ce)
            history['ppl'].append(val_ppl)
            history['acc'].append(val_acc)
            history['needle_acc'].append(needle_acc)
            history['stack_depth'].append(stack_depth)
            history['rule_loss'].append(train_rule)
            
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/PPL', val_ppl, epoch)
            self.writer.add_scalar('Val/Needle_Acc', needle_acc, epoch)
            self.writer.add_scalar('Val/Stack_Depth', stack_depth, epoch)
        
        self.save_results('stage0', history)
        return history
    
    def run_stage1(self):
        print("\n" + "="*60)
        print("STAGE 1: Add Stack + Gumbel Routing")
        print("="*60)
        
        dataloader = self.setup_stage1_curriculum()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=self.train_config.learning_rate * 0.5)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        history = {
            'loss': [], 'ce': [], 'ppl': [], 'acc': [], 'needle_acc': [],
            'stack_depth': [], 'rule_loss': []
        }
        
        for epoch in range(self.train_config.epochs):
            train_loss, train_ce, train_rule = self.train_epoch(dataloader, optimizer, criterion, epoch, '1')
            val_loss, val_ce, val_ppl, val_acc, needle_acc, stack_depth = self.evaluate(dataloader, criterion, '1')
            
            print(f"Epoch {epoch+1}/{self.train_config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"Needle Acc: {needle_acc:.4f} | Stack: {stack_depth:.2f}")
            
            history['loss'].append(val_loss)
            history['ce'].append(val_ce)
            history['ppl'].append(val_ppl)
            history['acc'].append(val_acc)
            history['needle_acc'].append(needle_acc)
            history['stack_depth'].append(stack_depth)
            history['rule_loss'].append(train_rule)
            
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/PPL', val_ppl, epoch)
            self.writer.add_scalar('Val/Needle_Acc', needle_acc, epoch)
            self.writer.add_scalar('Val/Stack_Depth', stack_depth, epoch)
            
            if epoch > 0 and epoch % 5 == 0:
                dataloader = self.setup_stage1_curriculum(epochs_trained=epoch)
        
        self.save_results('stage1', history)
        return history
    
    def run_stage2(self):
        print("\n" + "="*60)
        print("STAGE 2: Full Meta-Loss + Fault Traces")
        print("="*60)
        
        dataloader = self.setup_stage2_curriculum()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=self.train_config.learning_rate * 0.3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        history = {
            'loss': [], 'ce': [], 'ppl': [], 'acc': [], 'needle_acc': [],
            'stack_depth': [], 'rule_loss': []
        }
        
        for epoch in range(self.train_config.epochs):
            train_loss, train_ce, train_rule = self.train_epoch(dataloader, optimizer, criterion, epoch, '2')
            val_loss, val_ce, val_ppl, val_acc, needle_acc, stack_depth = self.evaluate(dataloader, criterion, '2')
            
            print(f"Epoch {epoch+1}/{self.train_config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"Needle Acc: {needle_acc:.4f} | Stack: {stack_depth:.2f}")
            
            history['loss'].append(val_loss)
            history['ce'].append(val_ce)
            history['ppl'].append(val_ppl)
            history['acc'].append(val_acc)
            history['needle_acc'].append(needle_acc)
            history['stack_depth'].append(stack_depth)
            history['rule_loss'].append(train_rule)
            
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/PPL', val_ppl, epoch)
            self.writer.add_scalar('Val/Needle_Acc', needle_acc, epoch)
            self.writer.add_scalar('Val/Stack_Depth', stack_depth, epoch)
        
        self.save_results('stage2', history)
        return history
    
    def save_results(self, stage, history):
        os.makedirs(self.train_config.output_dir, exist_ok=True)
        
        results_path = f'{self.train_config.output_dir}/results_{stage}_v2.json'
        with open(results_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if self.train_config.save_checkpoints:
            model_path = f'{self.train_config.output_dir}/model_{stage}_v2.pt'
            torch.save(self.model.state_dict(), model_path)
            print(f"Saved checkpoint to {model_path}")
        
        print(f"Results saved to {results_path}")
    
    def run_full_curriculum(self):
        print("\n" + "="*60)
        print("ANA V2 FULL TRAINING CURRICULUM")
        print("="*60)
        
        all_results = {}
        
        stage0_results = self.run_stage0()
        all_results['stage0'] = stage0_results
        
        stage1_results = self.run_stage1()
        all_results['stage1'] = stage1_results
        
        stage2_results = self.run_stage2()
        all_results['stage2'] = stage2_results
        
        combined_path = f'{self.train_config.output_dir}/results_full_curriculum_v2.json'
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.writer.close()
        print(f"\nFull curriculum complete. Results saved to {combined_path}")
        
        return all_results


import torch.nn.functional as F
