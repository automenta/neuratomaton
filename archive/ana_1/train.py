import torch
import torch.optim as optim
# from accelerate import Accelerator # Removed
# from transformers import get_scheduler
from config import ANAConfig, ANAMiniConfig
from model.modeling_ana import ANAModel
from data import get_dataloader
import time
import os
import sys

def main():
    # 1. Setup
    # accelerator = Accelerator(gradient_accumulation_steps=1)
    # device = accelerator.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Config & Model
    # Check args for mini
    if len(sys.argv) > 1 and sys.argv[1] == 'mini':
        print("Using Mini Config (10M)")
        config = ANAMiniConfig()
        BATCH_SIZE = 2 # Reduced from 8
        SEQ_LEN = 128  # Reduced from 512
        dataset_name = "wikitext"
    else:
        config = ANAConfig() # 125M defaults
        BATCH_SIZE = 4
        SEQ_LEN = 1024
        dataset_name = "slimpajama"

    model = ANAModel(config).to(device)
    
    # 3. Data
    dataloader = get_dataloader(BATCH_SIZE, SEQ_LEN, dataset_name=dataset_name)
    
    # 4. Opt
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 5. Loop
    model.train()
    step = 0
    log_interval = 10
    
    print(f"Starting ANA-Small (10M) Validation...")
    print(f"Top 5 Params: {[n for n, p in model.named_parameters()][:5]}")
    
    start_time = time.time()
    
    for batch in dataloader:
        # batch is [B, T+1] or dict depending on data.py
        # With wikitext/TensorDataset it returns tuple (input_ids,) or tensor?
        if isinstance(batch, list) or isinstance(batch, tuple):
             batch = batch[0] 
        elif isinstance(batch, dict):
             batch = batch['input_ids']
        
        batch = batch.to(device)
        
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # with accelerator.accumulate(model):
        if True:
            outputs = model(x, labels=y)
            loss = outputs['loss']
            
            # accelerator.backward(loss)
            loss.backward()
            
            # accelerator.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
        step += 1
        
        if step % log_interval == 0:
            loss_val = loss.item()
            others = outputs['others']
            # sparsity = others.get('spar', 0)
            
            elapsed = time.time() - start_time
            tps = (BATCH_SIZE * SEQ_LEN * log_interval) / elapsed
            start_time = time.time()
            
            print(f"Step {step} | Loss: {loss_val:.4f} | TPS: {tps:.0f} | Sparsity: {others.get('spar', 0.0):.4f}")
            
        if step >= max_steps:
            break
            
    print("Validation Complete. Model is runnable.")
    # accelerator.save_state("ana_small_ckpt")
    torch.save(model.state_dict(), "ana_mini.pt")

if __name__ == "__main__":
    main()
