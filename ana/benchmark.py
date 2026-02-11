"""
ANA Benchmark Suite - Systematic evaluation of algorithmic reasoning
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ana import ANAConfig, ANAModel
from ana.models import BaselineSSM
from ana.tasks import TASK_REGISTRY
import json


def collate_with_mask(batch):
    """Collate function that handles variable-length sequences and masks."""
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        masks = None
    else:
        xs, ys, masks = zip(*batch)
    
    max_len = max(x.size(0) for x in xs)
    
    xs_pad = torch.stack([F.pad(x, (0, max_len - x.size(0))) for x in xs])
    ys_pad = torch.stack([F.pad(y, (0, max_len - y.size(0)), value=-100) for y in ys])
    
    if masks is not None:
        masks_pad = torch.stack([F.pad(m, (0, max_len - m.size(0))) for m in masks])
        return xs_pad, ys_pad, masks_pad
    return xs_pad, ys_pad, None


def evaluate_generalization(
    model,
    task_name,
    train_lengths,
    test_lengths,
    vocab_size=20,
    steps_per_length=50,
    lr=1e-2,
    device='cpu'
):
    """
    Evaluate model's ability to generalize to longer sequences.
    
    Returns dict with:
        - train_accuracy: accuracy on training lengths
        - test_accuracy: accuracy on test lengths (generalization)
        - k_ratio: test_length / max(train_length)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    results = {'train': {}, 'test': {}}
    
    # Training
    model.train()
    for L in train_lengths:
        TaskClass = TASK_REGISTRY[task_name]
        dataset = TaskClass(num_samples=steps_per_length * 16, seq_len=L, vocab_size=vocab_size)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_with_mask)
        
        for x, y, mask in loader:
            x, y = x.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss_raw = criterion(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size())
            
            if mask is not None:
                loss = (loss_raw * mask).sum() / mask.sum()
            else:
                loss = loss_raw.mean()
            
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train accuracy
        for L in train_lengths:
            dataset = TaskClass(num_samples=100, seq_len=L, vocab_size=vocab_size)
            loader = DataLoader(dataset, batch_size=16, collate_fn=collate_with_mask)
            
            correct, total = 0, 0
            for x, y, mask in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(-1)
                
                valid = (y != -100)
                correct += (preds[valid] == y[valid]).sum().item()
                total += valid.sum().item()
            
            results['train'][L] = correct / total if total > 0 else 0
        
        # Test accuracy (generalization)
        max_train = max(train_lengths)
        for L in test_lengths:
            dataset = TaskClass(num_samples=100, seq_len=L, vocab_size=vocab_size)
            loader = DataLoader(dataset, batch_size=16, collate_fn=collate_with_mask)
            
            correct, total = 0, 0
            for x, y, mask in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(-1)
                
                valid = (y != -100)
                correct += (preds[valid] == y[valid]).sum().item()
                total += valid.sum().item()
            
            results['test'][L] = {
                'accuracy': correct / total if total > 0 else 0,
                'k_ratio': L / max_train
            }
    
    return results


def run_benchmark_suite(
    model_class,
    config,
    tasks=['copy', 'reverse', 'associative_recall'],
    train_lengths=[2, 3, 4, 5, 6],
    test_lengths=[7, 8, 10, 12],
    device='cpu'
):
    """Run complete benchmark suite on a model."""
    all_results = {}
    
    for task_name in tasks:
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        
        model = model_class(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        
        results = evaluate_generalization(
            model, task_name, train_lengths, test_lengths,
            vocab_size=config.vocab_size, device=device
        )
        
        all_results[task_name] = results
        
        print(f"\nTrain accuracy:")
        for L, acc in results['train'].items():
            print(f"  Length {L}: {100*acc:.1f}%")
        
        print(f"\nGeneralization:")
        for L, data in results['test'].items():
            print(f"  Length {L} (k={data['k_ratio']:.1f}): {100*data['accuracy']:.1f}%")
    
    return all_results


def compare_models(config, device='cpu'):
    """Compare ANA vs BaselineSSM on all tasks."""
    print("="*60)
    print("ANA vs BaselineSSM Comparison")
    print("="*60)
    
    results = {}
    
    for name, ModelClass in [('ANA', ANAModel), ('Baseline', BaselineSSM)]:
        print(f"\n--- {name} ---")
        results[name] = run_benchmark_suite(
            ModelClass, config,
            tasks=['copy', 'reverse'],
            device=device
        )
    
    return results


if __name__ == "__main__":
    config = ANAConfig(
        d_model=32, vocab_size=20, state_dim=32,
        track_count=2, num_layers=2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = compare_models(config, device)
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
